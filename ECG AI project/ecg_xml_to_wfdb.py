from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from code.service.xml_ecg import convert_xml_file_to_wfdb


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert an HL7 AnnotatedECG XML file to 12-lead 500 Hz WFDB and optional Inception-ready NPY."
    )
    parser.add_argument("xml", help="Input ECG XML file.")
    parser.add_argument(
        "--out-dir",
        default=str(root / "data" / "xml_to_wfdb_converted"),
        help="Output directory for WFDB files and manifest JSON.",
    )
    parser.add_argument("--record-name", default="", help="Safe output WFDB record name. Defaults to XML_UPLOAD_<sha12>.")
    parser.add_argument("--write-npy", action="store_true", help="Also write the fixed [12,5000] float32 waveform array.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    record_name = args.record_name.strip() or None
    result = convert_xml_file_to_wfdb(args.xml, out_dir, record_name=record_name)
    manifest = {
        "record_name": result.record_name,
        "wfdb_record_path": result.wfdb_record_path,
        "source_xml_sha256": result.source_xml_sha256,
        "input_metrics": result.input_metrics(),
    }
    if args.write_npy:
        npy_path = out_dir / f"{result.record_name}.npy"
        np.save(npy_path, result.waveform_uv_500hz.astype(np.float32, copy=False))
        manifest["npy_path"] = str(npy_path)
    manifest_path = out_dir / f"{result.record_name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
