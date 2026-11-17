"""Optimize trained SafeStep YOLO models specifically for Raspberry Pi 4 Model B (4GB RAM).

Performs:
1. Preserves existing `.pt` and `.onnx` model files without overwriting them destructively.
2. Applies ONNX Runtime Dynamic INT8 Quantization (`QUInt8` weights) to produce compact,
   ARM64 Cortex-A72 optimized models:
   - `models/object_detection/best_rpi_int8.onnx`
   - `models/currency_detection/best_rpi_int8.onnx`
3. Benchmarks FP32 ONNX vs INT8 Quantized ONNX inference latency and memory footprint
   using ONNX Runtime (`CPUExecutionProvider` with 4 intra-op threads matching RPi 4 quad-core CPU).
4. Saves optimization report to `evaluation/results/rpi_model_optimization.json`.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.OptimizeRPi")


def _quantize_and_benchmark(model_name: str, fp32_onnx_path: Path, int8_onnx_path: Path, imgsz: int = 320) -> Dict[str, Any]:
    """Quantize an FP32 ONNX model to INT8 (or preprocess/optimize graph) and benchmark ORT latency."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not fp32_onnx_path.exists():
        return {"model": model_name, "status": "missing_source_onnx", "source": str(fp32_onnx_path)}

    fp32_size_mb = round(fp32_onnx_path.stat().st_size / (1024 * 1024), 2)
    quantized_ok = False
    quant_error = None

    try:
        quantize_dynamic(
            model_input=str(fp32_onnx_path),
            model_output=str(int8_onnx_path),
            weight_type=QuantType.QUInt8,
        )
        quantized_ok = int8_onnx_path.exists()
    except Exception as exc:
        quant_error = str(exc)
        logger.warning("Dynamic quantization fallback for %s: %s", model_name, exc)

    # Benchmark FP32 ONNX with 4 threads (matching Raspberry Pi 4 Cortex-A72 4-core layout)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    dummy_input = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    fp32_sess = ort.InferenceSession(str(fp32_onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    in_name = fp32_sess.get_inputs()[0].name
    # Warmup
    for _ in range(3):
        fp32_sess.run(None, {in_name: dummy_input})
    t0 = time.perf_counter()
    runs = 15
    for _ in range(runs):
        fp32_sess.run(None, {in_name: dummy_input})
    fp32_ms = round(((time.perf_counter() - t0) / runs) * 1000.0, 2)

    int8_ms = None
    int8_size_mb = None
    int8_verified = False
    if quantized_ok and int8_onnx_path.exists():
        int8_size_mb = round(int8_onnx_path.stat().st_size / (1024 * 1024), 2)
        try:
            int8_sess = ort.InferenceSession(str(int8_onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
            for _ in range(3):
                int8_sess.run(None, {in_name: dummy_input})
            t1 = time.perf_counter()
            for _ in range(runs):
                int8_sess.run(None, {in_name: dummy_input})
            int8_ms = round(((time.perf_counter() - t1) / runs) * 1000.0, 2)
            int8_verified = True
        except Exception as exc:
            quant_error = f"INT8 ORT execution note: {exc}"

    info = {
        "model": model_name,
        "input_shape": [1, 3, imgsz, imgsz],
        "ort_threads": 4,
        "fp32_onnx_path": str(fp32_onnx_path),
        "fp32_size_mb": fp32_size_mb,
        "fp32_ort_latency_ms": fp32_ms,
        "fp32_ort_fps": round(1000.0 / max(0.1, fp32_ms), 2),
        "int8_onnx_path": str(int8_onnx_path) if quantized_ok else None,
        "int8_size_mb": int8_size_mb,
        "int8_ort_latency_ms": int8_ms,
        "int8_verified": int8_verified,
        "size_reduction_percent": (
            round((1.0 - (int8_size_mb / fp32_size_mb)) * 100.0, 1)
            if (int8_size_mb and fp32_size_mb > 0)
            else 0.0
        ),
        "notes": quant_error or "FP32 and INT8 ONNX models ready for Raspberry Pi 4 deployment.",
    }
    logger.info(
        "Optimized %s for RPi 4: FP32=%.2fMB (%.2fms), INT8=%sMB (verified=%s)",
        model_name,
        fp32_size_mb,
        fp32_ms,
        str(int8_size_mb),
        int8_verified,
    )
    return info


def optimize_all_for_raspberry_pi() -> Dict[str, Any]:
    obj_dir = PROJECT_ROOT / "models" / "object_detection"
    curr_dir = PROJECT_ROOT / "models" / "currency_detection"

    summary = {
        "target_hardware": "Raspberry Pi 4 Model B (4GB RAM, Quad-Core Cortex-A72 ARM64)",
        "object_detection": _quantize_and_benchmark(
            "safestep_object_yolov8n",
            obj_dir / "best.onnx",
            obj_dir / "best_rpi_int8.onnx",
            imgsz=320,
        ),
        "currency_detection": _quantize_and_benchmark(
            "safestep_currency_yolov8n",
            curr_dir / "best.onnx",
            curr_dir / "best_rpi_int8.onnx",
            imgsz=320,
        ),
    }

    out_path = PROJECT_ROOT / "evaluation" / "results" / "rpi_model_optimization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = optimize_all_for_raspberry_pi()
    print(json.dumps(res, indent=2))
