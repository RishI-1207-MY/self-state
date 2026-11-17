"""Export trained SafeStep YOLO models to ONNX (and TFLite where supported).

Verifies exported ONNX artifacts using `onnx` and `onnxruntime`.
Records exact export status and file sizes in `evaluation/results/exported_models.json`.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.Export")


def export_yolo_model(weights_path: Path, imgsz: int = 320, try_tflite: bool = False) -> Dict[str, Any]:
    """Export a trained YOLO `.pt` checkpoint to ONNX and verify with ONNX Runtime."""
    from ultralytics import YOLO

    if not weights_path.exists():
        return {"status": "failed", "reason": f"Weights not found: {weights_path}"}

    model = YOLO(str(weights_path))
    onnx_exported_path = model.export(format="onnx", imgsz=imgsz, simplify=False)
    onnx_path = Path(str(onnx_exported_path))
    target_onnx = weights_path.parent / "best.onnx"
    if onnx_path.exists() and onnx_path.resolve() != target_onnx.resolve():
        shutil.copy2(onnx_path, target_onnx)

    onnx_valid = False
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(target_onnx))
        onnx.checker.check_model(onnx_model)
        _ = ort.InferenceSession(str(target_onnx), providers=["CPUExecutionProvider"])
        onnx_valid = True
    except Exception as exc:
        logger.warning("ONNX verification warning for %s: %s", target_onnx, exc)

    tflite_status = "skipped (TensorFlow not installed in lightweight CPU environment)"
    if try_tflite:
        try:
            import tensorflow  # type: ignore # noqa: F401
            tf_out = model.export(format="tflite", imgsz=imgsz)
            tflite_status = f"exported: {tf_out}"
        except Exception as exc:
            tflite_status = f"unavailable: {exc}"

    info = {
        "source_pt": str(weights_path),
        "onnx_path": str(target_onnx),
        "onnx_exists": target_onnx.exists(),
        "onnx_verified_ort": onnx_valid,
        "onnx_size_mb": round(target_onnx.stat().st_size / (1024 * 1024), 2) if target_onnx.exists() else 0.0,
        "tflite_status": tflite_status,
    }
    logger.info("Exported %s -> ONNX verified=%s", weights_path.name, onnx_valid)
    return info


def export_all_models(imgsz: int = 320, try_tflite: bool = False) -> Dict[str, Any]:
    obj_pt = PROJECT_ROOT / "models" / "object_detection" / "best.pt"
    curr_pt = PROJECT_ROOT / "models" / "currency_detection" / "best.pt"

    summary = {
        "object_detection": export_yolo_model(obj_pt, imgsz=imgsz, try_tflite=try_tflite),
        "currency_detection": export_yolo_model(curr_pt, imgsz=imgsz, try_tflite=try_tflite),
    }
    out_file = PROJECT_ROOT / "evaluation" / "results" / "exported_models.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SafeStep YOLO models to ONNX / TFLite.")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--try-tflite", action="store_true")
    args = parser.parse_args()
    export_all_models(imgsz=args.imgsz, try_tflite=args.try_tflite)
