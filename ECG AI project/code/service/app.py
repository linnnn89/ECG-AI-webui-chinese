# -*- coding: utf-8 -*-
import json
import os
import re
import tempfile
import hashlib
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import wfdb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from code.casebank.features import extract_basic_wave_features
from code.casebank.io import CASEBANK_REQUIRED_FILES
from code.casebank.render import render_case_png, resolve_rendered_png
from code.casebank.search import CaseBankStore, SearchEngine

from .config import (
    CASEBANK_CACHE_DIR,
    CASEBANK_DIR,
    CASEBANK_SCORE_THRESHOLD,
    CASEBANK_TOP_K,
    CLASSES_JSON,
    DIGITIZER_BACKEND,
    MIN_DIGITIZED_FINITE_FRACTION,
    MIN_DIGITIZED_LEAD_COUNT,
    MIN_DIGITIZED_LEAD_FINITE_FRACTION,
    MODEL_DIR,
    ONNX_PATH,
    ROOT,
    TARGET_LEN,
    THRESHOLDS_JSON,
    UNCERTAINTY_ENABLED,
    UNCERTAINTY_LABEL,
    UNCERTAINTY_MAX_PROB,
    UNCERTAINTY_MESSAGE,
)
from .digitize import DigitiseError, backend_status, digitize_image
from .layout import Layout, detect_layout
from .xml_ecg import LEAD_ORDER, XmlConversionError, convert_xml_file_to_wfdb


APP = FastAPI(title="ECG Inference Service")

PTB = os.path.join(ROOT, "data", "ptbxl")
WFDB_R = os.path.join(PTB, "wfdb")

SESS = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)
with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
    THR_MAP = json.load(f)
THR = np.array([THR_MAP.get(c, 0.5) for c in CLASSES], dtype="float32")
CASEBANK_STORE = None
CASEBANK_LOAD_ERROR = None

ZH = {
    "NORM": "正常心电图",
    "MI": "心肌梗死相关改变",
    "STTC": "ST-T异常",
    "LVH": "左心室肥厚",
    "LBBB": "左束支传导阻滞",
    "RBBB": "右束支传导阻滞",
    "1AVB": "一度房室传导阻滞",
    "2AVB": "二度房室传导阻滞",
    "3AVB": "三度房室传导阻滞",
    "WPW": "预激综合征（WPW）",
    "UNKNOWN": "结果高度不确定，建议进一步参考评估",
}

SOURCE_ZH = {
    "wfdb": "WFDB记录",
    "wfdb_upload": "WFDB文件上传",
    "image": "图像上传",
    "xml": "XML上传",
}


def _safe_upload_basename(filename: str, fallback: str) -> str:
    name = os.path.basename(filename or "").strip()
    if not name:
        return fallback
    name = re.sub(r"[\x00-\x1f]+", "_", name)
    return name[:160] or fallback


def _canonical_lead_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def _wfdb_signal_to_model_signal(sig: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
    arr = np.asarray(sig, dtype="float32")
    if arr.ndim != 2:
        raise ValueError(f"Expected WFDB signal [T,C], got {arr.shape}")
    sig_names = list(info.get("sig_name") or [])
    qc_notes = []
    if len(sig_names) == arr.shape[1]:
        lead_index = {_canonical_lead_name(name): idx for idx, name in enumerate(sig_names)}
        wanted = [_canonical_lead_name(name) for name in LEAD_ORDER]
        if all(name in lead_index for name in wanted):
            arr = arr[:, [lead_index[name] for name in wanted]]
        elif arr.shape[1] >= 12:
            qc_notes.append("wfdb_lead_names_incomplete_used_first_12")
            arr = arr[:, :12]
        else:
            raise ValueError(f"WFDB record has {arr.shape[1]} leads; 12 leads are required")
    elif arr.shape[1] >= 12:
        qc_notes.append("wfdb_missing_signal_names_used_first_12")
        arr = arr[:, :12]
    else:
        raise ValueError(f"WFDB record has {arr.shape[1]} leads; 12 leads are required")

    x = arr.T.astype("float32", copy=False)
    metrics = {
        "n_leads": int(x.shape[0]),
        "original_fs_hz": float(info.get("fs") or 0.0),
        "original_num_samples": int(x.shape[1]),
        "finite_fraction": round(float(np.isfinite(x).sum() / x.size), 6) if x.size else 0.0,
        "lead_order": LEAD_ORDER,
        "qc_notes": ";".join(qc_notes),
    }
    return x, metrics


def _finalize_input_metrics(metrics: dict, model_x: np.ndarray) -> dict:
    out = dict(metrics or {})
    out.update(
        {
            "model_shape": [int(model_x.shape[0]), int(model_x.shape[1])],
            "model_num_samples": int(model_x.shape[1]),
        }
    )
    return out


def _norm_and_resample(x: np.ndarray, L: int = TARGET_LEN) -> np.ndarray:
    """每导联z标准化，并重采样到L。"""
    x = np.asarray(x, dtype="float32")
    if not np.isfinite(x).any():
        raise ValueError("signal contains no finite values")
    if np.isnan(x).any() or np.isinf(x).any():
        cleaned = x.copy()
        for c in range(cleaned.shape[0]):
            lead = cleaned[c]
            finite = np.isfinite(lead)
            if finite.any():
                fill = float(np.nanmedian(lead[finite]))
                lead[~finite] = fill
            else:
                lead[:] = 0.0
        x = cleaned
    C, n = x.shape
    if n != L:
        idx = np.linspace(0, n - 1, L)
        base = np.arange(n, dtype=float)
        y = np.zeros((C, L), dtype="float32")
        for c in range(C):
            y[c] = np.interp(idx, base, x[c])
        x = y
    m = x.mean(1, keepdims=True)
    s = x.std(1, keepdims=True) + 1e-6
    return (x - m) / s


def _predict_prob(x: np.ndarray) -> np.ndarray:
    logits = SESS.run(None, {"x": x[None, ...]})[0][0]
    return 1.0 / (1.0 + np.exp(-logits))


def _classify_prob(prob: np.ndarray) -> dict:
    triggered_raw = {c: bool(p >= t) for c, p, t in zip(CLASSES, prob, THR)}
    triggered = dict(triggered_raw)
    pathology = [c for c in CLASSES if c != "NORM" and triggered_raw[c]]
    max_idx = int(np.argmax(prob))
    max_prob = float(prob[max_idx])
    max_class = CLASSES[max_idx]
    low_confidence = max_prob < UNCERTAINTY_MAX_PROB
    norm_exclusion_applied = False

    if pathology:
        if triggered_raw.get("NORM", False):
            triggered["NORM"] = False
            norm_exclusion_applied = True
        return {
            "labels": pathology,
            "uncertain": False,
            "uncertainty_reason": None,
            "low_confidence": False,
            "max_prob": max_prob,
            "max_prob_class": max_class,
            "message": None,
            "triggered": triggered,
            "triggered_raw": triggered_raw,
            "norm_exclusion_applied": norm_exclusion_applied,
        }

    if triggered_raw["NORM"]:
        return {
            "labels": ["NORM"],
            "uncertain": False,
            "uncertainty_reason": None,
            "low_confidence": False,
            "max_prob": max_prob,
            "max_prob_class": max_class,
            "message": None,
            "triggered": triggered,
            "triggered_raw": triggered_raw,
            "norm_exclusion_applied": False,
        }

    if UNCERTAINTY_ENABLED:
        reason = "low_confidence_all_classes" if low_confidence else "all_classes_below_threshold"
        return {
            "labels": [UNCERTAINTY_LABEL],
            "uncertain": True,
            "uncertainty_reason": reason,
            "low_confidence": low_confidence,
            "max_prob": max_prob,
            "max_prob_class": max_class,
            "message": UNCERTAINTY_MESSAGE,
            "triggered": triggered,
            "triggered_raw": triggered_raw,
            "norm_exclusion_applied": False,
        }

    return {
        "labels": [UNCERTAINTY_LABEL],
        "uncertain": False,
        "uncertainty_reason": "disabled",
        "low_confidence": low_confidence,
        "max_prob": max_prob,
        "max_prob_class": max_class,
        "message": None,
        "triggered": triggered,
        "triggered_raw": triggered_raw,
        "norm_exclusion_applied": False,
    }


def _casebank_status() -> dict:
    missing = [name for name in CASEBANK_REQUIRED_FILES if not os.path.exists(os.path.join(CASEBANK_DIR, name))]
    config = {}
    if not missing:
        config_path = os.path.join(CASEBANK_DIR, "build_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except OSError:
            config = {}
    return {
        "enabled": not missing,
        "casebank_dir": CASEBANK_DIR,
        "cache_dir": CASEBANK_CACHE_DIR,
        "top_k": CASEBANK_TOP_K,
        "score_threshold": CASEBANK_SCORE_THRESHOLD,
        "missing_files": missing,
        "num_cases": config.get("num_cases"),
        "source_counts": config.get("source_counts", {}),
        "build_version": config.get("build_version"),
        "load_error": CASEBANK_LOAD_ERROR,
    }


def _get_casebank_store():
    global CASEBANK_STORE, CASEBANK_LOAD_ERROR
    if CASEBANK_STORE is not None:
        return CASEBANK_STORE
    try:
        store = CaseBankStore.load(CASEBANK_DIR)
    except Exception as exc:  # noqa: BLE001 - health/result should report cleanly.
        CASEBANK_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    CASEBANK_LOAD_ERROR = None
    CASEBANK_STORE = store
    return store


def _casebank_payload(record_id: str, prob: np.ndarray, model_x: np.ndarray | None) -> dict:
    store = _get_casebank_store()
    if store is None:
        status = _casebank_status()
        return {**status, "matches": [], "warnings": ["CaseBank index is not available."]}
    if list(store.classes) != list(CLASSES):
        return {
            **_casebank_status(),
            "enabled": False,
            "matches": [],
            "warnings": ["CaseBank classes do not match the active model classes."],
        }
    if model_x is None:
        return {**_casebank_status(), "matches": [], "warnings": ["Model input tensor was not supplied for CaseBank."]}

    margins = np.asarray(prob, dtype=np.float32) - THR
    wave_features = extract_basic_wave_features(model_x)
    result = SearchEngine(store).search(
        probabilities=prob,
        margins=margins,
        wave_features=wave_features,
        predicted_labels=[],
        query_case_id=record_id,
        query_record_path=None,
        top_k=CASEBANK_TOP_K,
        prefetch_k=300,
        min_candidates=50,
        score_threshold=CASEBANK_SCORE_THRESHOLD,
        ablation="pure_vector",
    )
    matches = []
    for item in result.similar_cases:
        matches.append(
            {
                "rank": item.rank,
                "case_id": item.case_id,
                "source": item.source,
                "score": item.score,
                "score_level": item.score_level,
                "true_diagnosis": item.true_diagnosis or "|".join(item.labels) or "unmapped",
                "label_scope": item.label_scope,
                "labels": item.labels,
                "predicted_labels": item.predicted_labels,
                "probabilities": item.probabilities,
                "image_url": item.image_url,
            }
        )
    return {**_casebank_status(), "matches": matches, "warnings": result.warnings}


def _image_error_detail(error_code: str, message: str, suggestion: str, **technical_detail):
    return {
        "error_code": error_code,
        "message": message,
        "suggestion": suggestion,
        "technical_detail": technical_detail,
    }


def _report_dict(
    record_id: str,
    prob: np.ndarray,
    source: str,
    layout: str,
    input_metrics: dict | None = None,
    model_x: np.ndarray | None = None,
):
    print(f"\n========== ECG inference: {record_id} ({source}) ==========")
    for c, p, t in zip(CLASSES, prob, THR):
        trigger = " <--- triggered" if p >= t else ""
        print(f"[{c}] prob: {p:.4f} | threshold: {t:.4f}{trigger}")
    print("====================================================\n")

    decision = _classify_prob(prob)
    pos = decision["labels"]
    zh = [ZH.get(c, c) for c in pos]
    message = decision["message"]
    note = f"\n安全提示: {message}" if message else ""
    norm_note = "\n互斥规则: 已移除与异常诊断共存的NORM触发。" if decision["norm_exclusion_applied"] else ""
    source_zh = SOURCE_ZH.get(source, source)
    return {
        "source": source,
        "source_zh": source_zh,
        "layout": layout,
        "record": record_id,
        "model_dir": MODEL_DIR,
        "classes": CLASSES,
        "prob": {c: float(p) for c, p in zip(CLASSES, prob)},
        "thresholds": {c: float(t) for c, t in zip(CLASSES, THR)},
        "labels": pos,
        "labels_zh": zh,
        "uncertain": decision["uncertain"],
        "uncertainty_reason": decision["uncertainty_reason"],
        "low_confidence": decision["low_confidence"],
        "max_prob": decision["max_prob"],
        "max_prob_class": decision["max_prob_class"],
        "uncertainty_message": message,
        "triggered": decision["triggered"],
        "triggered_raw": decision["triggered_raw"],
        "norm_exclusion_applied": decision["norm_exclusion_applied"],
        "input_metrics": input_metrics or {},
        "casebank": _casebank_payload(record_id, prob, model_x),
        "text": (
            f"心电图自动分析报告（{source_zh}）\n"
            f"记录: {record_id}\n"
            f"结论: {'、'.join(zh)}\n"
            f"说明: 本报告基于研究模型，需结合临床。{note}{norm_note}"
        ),
    }


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "classes": CLASSES,
        "model_dir": MODEL_DIR,
        "onnx_path": ONNX_PATH,
        "classes_json": CLASSES_JSON,
        "thresholds_json": THRESHOLDS_JSON,
        "thresholds": {c: float(t) for c, t in zip(CLASSES, THR)},
        "digitizer": backend_status(),
        "digitized_quality_gate": {
            "min_finite_fraction": MIN_DIGITIZED_FINITE_FRACTION,
            "min_lead_finite_fraction": MIN_DIGITIZED_LEAD_FINITE_FRACTION,
            "min_lead_count": MIN_DIGITIZED_LEAD_COUNT,
        },
        "input_modes": {
            "wfdb_upload": True,
            "image_upload": True,
            "xml_upload": True,
            "xml_converter": "xml_to_wfdb_to_inception",
        },
        "uncertainty": {
            "enabled": UNCERTAINTY_ENABLED,
            "max_prob": UNCERTAINTY_MAX_PROB,
            "label": UNCERTAINTY_LABEL,
            "message": UNCERTAINTY_MESSAGE,
        },
        "casebank": _casebank_status(),
    }


@APP.get("/digitizer_status")
def digitizer_status():
    return backend_status()


class InferReq(BaseModel):
    record: str


@APP.post("/infer_record")
def infer_record(req: InferReq):
    rec = os.path.splitext(os.path.join(WFDB_R, req.record))[0]
    sig, info = wfdb.rdsamp(rec)
    raw_x, metrics = _wfdb_signal_to_model_signal(sig, info)
    x = _norm_and_resample(raw_x)
    metrics = _finalize_input_metrics(metrics, x)
    prob = _predict_prob(x)
    return _report_dict(os.path.basename(req.record), prob, "wfdb", "n/a", metrics, x)


@APP.post("/infer_wfdb_files")
async def infer_wfdb_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(
            status_code=400,
            detail=_image_error_detail(
                "WFDB_FILES_MISSING",
                "未收到WFDB文件。",
                "请选择同一条记录的 .hea 文件以及对应的 .dat 或 .mat 数据文件。",
            ),
        )

    allowed = {".hea", ".dat", ".mat"}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        saved_files = []
        hea_paths = []
        for idx, upload in enumerate(files, start=1):
            suffix = os.path.splitext(upload.filename or "")[1].lower()
            if suffix not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=_image_error_detail(
                        "WFDB_FILE_TYPE_UNSUPPORTED",
                        "当前文件格式不支持，无法作为WFDB记录读取。",
                        "请上传 .hea 以及同名 .dat 或 .mat 文件。",
                        suffix=suffix,
                    ),
                )
            safe_name = _safe_upload_basename(upload.filename or "", f"upload_{idx}{suffix}")
            out_path = tmp_path / safe_name
            out_path.write_bytes(await upload.read())
            saved_files.append(out_path)
            if suffix == ".hea":
                hea_paths.append(out_path)

        if not hea_paths:
            raise HTTPException(
                status_code=400,
                detail=_image_error_detail(
                    "WFDB_HEADER_MISSING",
                    "未找到WFDB头文件。",
                    "请选择 .hea 文件，并同时选择对应的数据文件。",
                    uploaded_files=len(saved_files),
                ),
            )

        hea_path = hea_paths[0]
        try:
            sig, info = wfdb.rdsamp(str(hea_path.with_suffix("")))
            raw_x, metrics = _wfdb_signal_to_model_signal(sig, info)
        except Exception as exc:  # noqa: BLE001 - expose clean WebUI error.
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "WFDB_READ_FAILED",
                    "WFDB文件无法读取或导联不完整。",
                    "请确认 .hea 与对应 .dat/.mat 来自同一条记录，并且包含完整12导联。",
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            ) from exc

        x = _norm_and_resample(raw_x)
        metrics = _finalize_input_metrics(
            {
                **metrics,
                "uploaded_file_count": len(saved_files),
                "wfdb_header_seen": True,
            },
            x,
        )
        prob = _predict_prob(x)
        try:
            digest = hashlib.sha256(hea_path.read_bytes()).hexdigest()[:12]
        except OSError:
            digest = "record"
        record_id = f"WFDB_UPLOAD_{digest[:12] or 'record'}"
        return _report_dict(record_id, prob, "wfdb_upload", "uploaded_wfdb", metrics, x)


@APP.post("/infer_xml")
async def infer_xml(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix != ".xml":
        raise HTTPException(
            status_code=400,
            detail=_image_error_detail(
                "XML_FILE_TYPE_UNSUPPORTED",
                "当前文件格式不支持，无法作为心电图XML读取。",
                "请上传 .xml 格式的HL7 Annotated ECG文件。",
                suffix=suffix,
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail=_image_error_detail(
                "XML_FILE_EMPTY",
                "上传的XML文件为空。",
                "请重新选择有效的心电图XML文件。",
            ),
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        xml_path = tmp_path / "upload.xml"
        xml_path.write_bytes(data)
        record_name = f"XML_UPLOAD_{hashlib.sha256(data).hexdigest()[:12]}"
        try:
            conversion = convert_xml_file_to_wfdb(xml_path, tmp_path / "wfdb", record_name=record_name)
            sig, info = wfdb.rdsamp(conversion.wfdb_record_path)
            raw_x, metrics = _wfdb_signal_to_model_signal(sig, info)
        except XmlConversionError as exc:
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "XML_CONVERSION_FAILED",
                    "XML文件无法转换为模型可用的12导联心电图。",
                    "请确认该XML是包含12导联波形的HL7 Annotated ECG格式。",
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            ) from exc
        except Exception as exc:  # noqa: BLE001 - return a clean API error.
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "XML_TO_WFDB_INFERENCE_FAILED",
                    "XML已进入转换流程，但WFDB中转或模型前处理失败。",
                    "请确认XML波形字段完整；如反复失败，请先用根目录转换工具单独检查。",
                    error_type=type(exc).__name__,
                    error=str(exc),
                ),
            ) from exc

        x = _norm_and_resample(raw_x)
        metrics = _finalize_input_metrics({**metrics, **conversion.input_metrics()}, x)
        prob = _predict_prob(x)
        return _report_dict(conversion.record_name, prob, "xml", "xml_to_wfdb", metrics, x)


@APP.post("/infer_image")
async def infer_image(file: UploadFile = File(...), layout: str = Form("auto")):
    if DIGITIZER_BACKEND == "none":
        raise HTTPException(
            status_code=501,
            detail=_image_error_detail(
                "IMAGE_DIGITIZER_DISABLED",
                "当前尚未启用心电图图片识别后端，因此无法分析上传图片。",
                "请先使用 WFDB/数字 ECG 记录进行分析；如需图片链路，请设置 ECG_DIGITIZER_BACKEND=felix 或 ahus。",
                configured_backend=DIGITIZER_BACKEND,
            ),
        )

    suf = os.path.splitext(file.filename or "")[1].lower()
    if suf not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        raise HTTPException(
            status_code=400,
            detail=_image_error_detail(
                "IMAGE_FILE_TYPE_UNSUPPORTED",
                "当前文件格式不支持，无法作为心电图图片识别。",
                "请上传 PNG、JPG、JPEG、BMP、TIF 或 TIFF 格式的心电图图片。",
                filename=file.filename,
                suffix=suf,
            ),
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
    try:
        tmp.write(await file.read())
        tmp.close()

        if layout == "auto" and DIGITIZER_BACKEND == "ahus":
            lay = Layout.UNK
        elif layout == "auto":
            lay = detect_layout(tmp.name)
        elif layout == "3x4":
            lay = Layout.L3x4
        elif layout == "6x6":
            lay = Layout.L6x6
        else:
            lay = Layout.UNK

        if lay == Layout.UNK and DIGITIZER_BACKEND not in {"ahus", "felix"}:
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "IMAGE_LAYOUT_UNRECOGNIZED",
                    "心电图图片版式无法正确识别，可能由于图片清晰度不足、拍摄角度偏斜、导联显示不完整或版式不符合要求导致。",
                    "请重新上传更清晰、完整的 12 导联心电图图片；也可以手动选择 3x4 版式后重试，或改用 WFDB/数字 ECG 文件。",
                    filename=file.filename,
                    requested_layout=layout,
                ),
            )
        if lay == Layout.L6x6 and DIGITIZER_BACKEND != "ahus":
            raise HTTPException(
                status_code=501,
                detail=_image_error_detail(
                    "IMAGE_LAYOUT_UNSUPPORTED",
                    "当前服务暂不支持 6+6 心电图版式的图片诊断。",
                    "请上传完整 3x4 版式的 12 导联心电图图片，或改用 WFDB/数字 ECG 文件。",
                    filename=file.filename,
                    detected_layout=lay.value,
                ),
            )

        try:
            x = digitize_image(tmp.name)
        except DigitiseError as exc:
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "IMAGE_DIGITIZATION_FAILED",
                    "心电图图片无法正确识别，可能由于清晰度不足、网格/波形对比度较低、导联显示不完整或图片版式不适配导致。",
                    "请重新上传更清晰、完整、无遮挡的 12 导联心电图图片；若仍失败，请改用 WFDB/数字 ECG 文件。",
                    filename=file.filename,
                    digitizer_backend=DIGITIZER_BACKEND,
                    error=str(exc),
                ),
            ) from exc

        finite_fraction = float(np.isfinite(x).sum() / x.size) if x.size else 0.0
        if finite_fraction < MIN_DIGITIZED_FINITE_FRACTION:
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "IMAGE_DIGITIZATION_LOW_QUALITY",
                    "心电图图片无法正确识别，可能由于图片清晰度不足、拍摄角度偏斜、导联显示不完整、版式不符合要求或扫描质量较差导致。",
                    "请重新上传更清晰、完整的 12 导联心电图图片，或改用 WFDB/数字 ECG 文件。",
                    filename=file.filename,
                    digitizer_backend=DIGITIZER_BACKEND,
                    finite_fraction=round(finite_fraction, 4),
                    required_finite_fraction=round(MIN_DIGITIZED_FINITE_FRACTION, 4),
                ),
            )

        lead_fractions = np.isfinite(x).mean(axis=1) if x.ndim == 2 and x.shape[0] == 12 else np.array([])
        covered_leads = int((lead_fractions >= MIN_DIGITIZED_LEAD_FINITE_FRACTION).sum())
        if covered_leads < MIN_DIGITIZED_LEAD_COUNT:
            raise HTTPException(
                status_code=422,
                detail=_image_error_detail(
                    "IMAGE_DIGITIZATION_INCOMPLETE_LEADS",
                    "心电图图片未能稳定提取完整 12 导联，因此暂不进入诊断模型。",
                    "请上传完整、无遮挡、导联显示清楚的 12 导联心电图图片，或改用 WFDB/数字 ECG 文件。",
                    filename=file.filename,
                    digitizer_backend=DIGITIZER_BACKEND,
                    covered_leads=covered_leads,
                    required_leads=MIN_DIGITIZED_LEAD_COUNT,
                    min_lead_finite_fraction=round(
                        float(lead_fractions.min()) if lead_fractions.size else 0.0,
                        4,
                    ),
                    median_lead_finite_fraction=round(
                        float(np.median(lead_fractions)) if lead_fractions.size else 0.0,
                        4,
                    ),
                    required_lead_finite_fraction=round(MIN_DIGITIZED_LEAD_FINITE_FRACTION, 4),
                ),
            )

        x = _norm_and_resample(x)
        prob = _predict_prob(x)
        report_layout = lay.value if lay != Layout.UNK else f"auto/{DIGITIZER_BACKEND}"
        return _report_dict(os.path.basename(file.filename), prob, "image", report_layout, None, x)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@APP.get("/casebank_image/{case_id}")
def casebank_image(case_id: str):
    store = _get_casebank_store()
    if store is None:
        raise HTTPException(status_code=404, detail="CaseBank index is not available.")
    case = next((item for item in store.cases if item.case_id == case_id), None)
    if case is None:
        raise HTTPException(status_code=404, detail="CaseBank case_id not found.")
    try:
        png_path = render_case_png(case, CASEBANK_CACHE_DIR)
        resolved = resolve_rendered_png(png_path)
    except Exception as exc:  # noqa: BLE001 - surface image rendering failure cleanly.
        raise HTTPException(status_code=422, detail=f"CaseBank image render failed: {type(exc).__name__}: {exc}") from exc
    return FileResponse(resolved, media_type="image/png", filename=f"{case.case_id}.png")


APP.mount(
    "/ui",
    StaticFiles(directory=os.path.join(ROOT, "code", "service", "ui"), html=True),
    name="ui",
)


@APP.get("/")
def _root():
    return RedirectResponse(url="/ui/")
