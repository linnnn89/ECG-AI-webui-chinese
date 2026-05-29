from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb


LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 500.0
TARGET_SECONDS = 10.0
TARGET_LEN = int(TARGET_FS * TARGET_SECONDS)

_LEAD_CODE_TO_NAME = {
    "MDC_ECG_LEAD_I": "I",
    "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III",
    "MDC_ECG_LEAD_AVR": "aVR",
    "MDC_ECG_LEAD_aVR": "aVR",
    "MDC_ECG_LEAD_AVL": "aVL",
    "MDC_ECG_LEAD_aVL": "aVL",
    "MDC_ECG_LEAD_AVF": "aVF",
    "MDC_ECG_LEAD_aVF": "aVF",
    "MDC_ECG_LEAD_V1": "V1",
    "MDC_ECG_LEAD_V2": "V2",
    "MDC_ECG_LEAD_V3": "V3",
    "MDC_ECG_LEAD_V4": "V4",
    "MDC_ECG_LEAD_V5": "V5",
    "MDC_ECG_LEAD_V6": "V6",
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "aVR",
    "aVR": "aVR",
    "AVL": "aVL",
    "aVL": "aVL",
    "AVF": "aVF",
    "aVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


class XmlConversionError(ValueError):
    pass


@dataclass
class ParsedXml:
    waveform_uv: np.ndarray
    original_fs_hz: float
    original_num_samples: int
    scale_units: List[str] = field(default_factory=list)
    qc_notes: List[str] = field(default_factory=list)


@dataclass
class XmlWfdbConversion:
    wfdb_record_path: str
    record_name: str
    waveform_uv_500hz: np.ndarray
    source_xml_sha256: str
    original_fs_hz: float
    original_num_samples: int
    converted_fs_hz: float
    converted_num_samples: int
    n_leads: int
    qc_notes: List[str]
    window_policy: str
    crop_left_samples_original_fs: int
    crop_right_samples_original_fs: int
    pad_left_samples_original_fs: int
    pad_right_samples_original_fs: int
    resample_method: str

    def input_metrics(self) -> Dict[str, object]:
        finite_fraction = float(np.isfinite(self.waveform_uv_500hz).sum() / self.waveform_uv_500hz.size)
        return {
            "n_leads": self.n_leads,
            "original_fs_hz": round(float(self.original_fs_hz), 6),
            "original_num_samples": int(self.original_num_samples),
            "converted_fs_hz": round(float(self.converted_fs_hz), 6),
            "converted_num_samples": int(self.converted_num_samples),
            "model_shape": [int(self.waveform_uv_500hz.shape[0]), int(self.waveform_uv_500hz.shape[1])],
            "finite_fraction": round(finite_fraction, 6),
            "wfdb_record": self.record_name,
            "window_policy": self.window_policy,
            "resample_method": self.resample_method,
            "qc_notes": ";".join(self.qc_notes),
        }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def _first_descendant(el: ET.Element, name: str) -> Optional[ET.Element]:
    for child in el.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _descendant_codes(el: ET.Element) -> List[str]:
    return [_text(child.attrib.get("code")) for child in el.iter() if _local_name(child.tag) == "code"]


def _lead_from_code(code: str) -> Optional[str]:
    if code in _LEAD_CODE_TO_NAME:
        return _LEAD_CODE_TO_NAME[code]
    tail = code.rsplit("_", 1)[-1]
    return _LEAD_CODE_TO_NAME.get(tail) or _LEAD_CODE_TO_NAME.get(tail.upper())


def _parse_float_attr(el: Optional[ET.Element], attr: str, default: float = 0.0) -> float:
    if el is None:
        return default
    raw = _text(el.attrib.get(attr))
    return float(raw) if raw else default


def _unit_scale_to_uv(unit: str) -> float:
    normalized = unit.replace("µ", "u").strip().lower()
    if normalized in {"uv", "microvolt", "microvolts"}:
        return 1.0
    if normalized in {"mv", "millivolt", "millivolts"}:
        return 1000.0
    if normalized in {"v", "volt", "volts"}:
        return 1_000_000.0
    return 1.0


def _time_to_seconds(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return value
    if normalized in {"ms", "msec", "millisecond", "milliseconds"}:
        return value / 1000.0
    if normalized in {"us", "usec", "microsecond", "microseconds"}:
        return value / 1_000_000.0
    return value


def _extract_time_increment_s(seq: ET.Element) -> Optional[float]:
    for name in ("increment", "scale"):
        el = _first_descendant(seq, name)
        if el is None:
            continue
        value = _parse_float_attr(el, "value", 0.0)
        unit = _text(el.attrib.get("unit"))
        seconds = _time_to_seconds(value, unit)
        if seconds > 0:
            return seconds
    return None


def _digits_to_array(text: str) -> np.ndarray:
    cleaned = re.sub(r"[,;\t\r\n]+", " ", text or "")
    arr = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    if arr.size == 0:
        raise XmlConversionError("Empty ECG digits field")
    return arr


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_annotated_ecg_xml(xml_path: Path) -> ParsedXml:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise XmlConversionError(f"Invalid XML: {exc}") from exc

    sequences = [el for el in root.iter() if _local_name(el.tag) == "sequence"]
    if not sequences:
        raise XmlConversionError("No ECG sequence elements found")

    time_increment_s: Optional[float] = None
    lead_values: Dict[str, np.ndarray] = {}
    scale_units: List[str] = []
    qc_notes: List[str] = []

    for seq in sequences:
        codes = _descendant_codes(seq)
        if any(code == "TIME_RELATIVE" or code.endswith("TIME_RELATIVE") for code in codes):
            time_increment_s = _extract_time_increment_s(seq) or time_increment_s
            continue

        lead = None
        for code in codes:
            lead = _lead_from_code(code)
            if lead:
                break
        if not lead:
            continue

        digits_el = _first_descendant(seq, "digits")
        if digits_el is None or not _text(digits_el.text):
            raise XmlConversionError(f"Missing digits for lead {lead}")
        digits = _digits_to_array(digits_el.text or "")
        origin_el = _first_descendant(seq, "origin")
        scale_el = _first_descendant(seq, "scale")
        origin = _parse_float_attr(origin_el, "value", 0.0)
        scale = _parse_float_attr(scale_el, "value", 1.0)
        unit = _text(scale_el.attrib.get("unit")) if scale_el is not None else ""
        scale_units.append(unit)
        lead_values[lead] = (origin + scale * digits) * _unit_scale_to_uv(unit)

    missing = [lead for lead in LEAD_ORDER if lead not in lead_values]
    if missing:
        raise XmlConversionError(f"Missing ECG leads: {', '.join(missing)}")
    if time_increment_s is None or time_increment_s <= 0:
        raise XmlConversionError("Missing or invalid TIME_RELATIVE increment")

    lengths = [len(lead_values[lead]) for lead in LEAD_ORDER]
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len <= 0:
        raise XmlConversionError("Empty ECG waveform")
    if min_len != max_len:
        qc_notes.append(f"lead_length_mismatch_min_{min_len}_max_{max_len};truncated_to_min")

    waveform = np.stack([lead_values[lead][:min_len] for lead in LEAD_ORDER], axis=0).astype(np.float32)
    if not np.isfinite(waveform).any():
        raise XmlConversionError("ECG waveform contains no finite values")

    return ParsedXml(
        waveform_uv=waveform,
        original_fs_hz=float(1.0 / time_increment_s),
        original_num_samples=int(min_len),
        scale_units=scale_units,
        qc_notes=qc_notes,
    )


def centered_10s_500hz(waveform_uv: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, Dict[str, object]]:
    waveform = np.asarray(waveform_uv, dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[0] != len(LEAD_ORDER):
        raise XmlConversionError(f"Expected waveform [12,T], got {waveform.shape}")
    if fs_hz <= 0:
        raise XmlConversionError(f"Invalid sampling rate: {fs_hz}")

    target_original = max(1, int(round(TARGET_SECONDS * fs_hz)))
    n = waveform.shape[1]
    pad_left = pad_right = 0
    crop_left = crop_right = 0
    if n >= target_original:
        start = (n - target_original) // 2
        cropped = waveform[:, start : start + target_original]
        crop_left = start
        crop_right = n - (start + target_original)
    else:
        pad_total = target_original - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        cropped = np.pad(waveform, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)

    if cropped.shape[1] == TARGET_LEN:
        converted = cropped.astype(np.float32, copy=False)
        resample_method = "none"
    else:
        base = np.arange(cropped.shape[1], dtype=np.float64)
        target = np.linspace(0, cropped.shape[1] - 1, TARGET_LEN)
        converted = np.zeros((cropped.shape[0], TARGET_LEN), dtype=np.float32)
        for idx in range(cropped.shape[0]):
            converted[idx] = np.interp(target, base, cropped[idx]).astype(np.float32)
        resample_method = "linear_interpolation"

    return converted, {
        "window_policy": "centered_10s",
        "crop_left_samples_original_fs": int(crop_left),
        "crop_right_samples_original_fs": int(crop_right),
        "pad_left_samples_original_fs": int(pad_left),
        "pad_right_samples_original_fs": int(pad_right),
        "resample_method": resample_method,
    }


def _safe_record_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\\-]+", "_", value).strip("_")
    return cleaned[:64] or "XML_UPLOAD"


def write_wfdb_record(out_dir: Path, record_name: str, waveform_uv: np.ndarray) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_record_name(record_name)
    wfdb.wrsamp(
        record_name=safe_name,
        fs=TARGET_FS,
        units=["uV"] * len(LEAD_ORDER),
        sig_name=LEAD_ORDER,
        p_signal=waveform_uv.T.astype(np.float64),
        fmt=["16"] * len(LEAD_ORDER),
        write_dir=str(out_dir),
    )
    return str(out_dir / safe_name)


def convert_xml_file_to_wfdb(
    xml_path: str | Path,
    out_dir: str | Path,
    record_name: Optional[str] = None,
) -> XmlWfdbConversion:
    xml_path = Path(xml_path)
    out_dir = Path(out_dir)
    parsed = parse_annotated_ecg_xml(xml_path)
    converted, window_meta = centered_10s_500hz(parsed.waveform_uv, parsed.original_fs_hz)
    source_hash = _sha256_file(xml_path)
    safe_record = _safe_record_name(record_name or f"XML_UPLOAD_{source_hash[:12]}")
    wfdb_record_path = write_wfdb_record(out_dir, safe_record, converted)
    qc_notes = list(parsed.qc_notes)
    if any(unit and unit.replace("µ", "u").strip().lower() not in {"uv", "mv", "v"} for unit in parsed.scale_units):
        qc_notes.append("unknown_scale_unit_seen")
    return XmlWfdbConversion(
        wfdb_record_path=wfdb_record_path,
        record_name=safe_record,
        waveform_uv_500hz=converted,
        source_xml_sha256=source_hash,
        original_fs_hz=parsed.original_fs_hz,
        original_num_samples=parsed.original_num_samples,
        converted_fs_hz=TARGET_FS,
        converted_num_samples=TARGET_LEN,
        n_leads=len(LEAD_ORDER),
        qc_notes=qc_notes,
        window_policy=str(window_meta["window_policy"]),
        crop_left_samples_original_fs=int(window_meta["crop_left_samples_original_fs"]),
        crop_right_samples_original_fs=int(window_meta["crop_right_samples_original_fs"]),
        pad_left_samples_original_fs=int(window_meta["pad_left_samples_original_fs"]),
        pad_right_samples_original_fs=int(window_meta["pad_right_samples_original_fs"]),
        resample_method=str(window_meta["resample_method"]),
    )

