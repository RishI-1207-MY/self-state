"""Evaluation script for SafeStep Indian Currency Detection Model.

Computes actual Precision, Recall, mAP@50, mAP@50-95, per-denomination performance,
and confusion matrix heatmap on the held-out test split.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.currency_detection import CURRENCY_CLASS_NAMES
from utils.logger import get_logger

logger = get_logger("SafeStep.Eval.Currency")


def evaluate_currency_model(
    weights_path: Path | None = None,
    data_yaml: Path | None = None,
    split: str = "test",
    img_size: int = 320,
) -> Dict[str, Any]:
    """Run Ultralytics validation on held-out currency test split and save actual metrics."""
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = PROJECT_ROOT / "models" / "currency_detection" / "best.pt"
    if data_yaml is None:
        data_yaml = PROJECT_ROOT / "datasets" / "currency_detection" / "data.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained currency detection weights not found at {weights_path}")

    results_dir = PROJECT_ROOT / "evaluation" / "results"
    graphs_dir = PROJECT_ROOT / "evaluation" / "graphs"
    for d in (results_dir, graphs_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=img_size,
        plots=True,
        project=str(results_dir),
        name="currency_eval_run",
        exist_ok=True,
        verbose=False,
    )
    eval_time_s = round(time.perf_counter() - t0, 2)

    box_metrics = metrics.box
    precision = round(float(box_metrics.mp), 4)
    recall = round(float(box_metrics.mr), 4)
    map50 = round(float(box_metrics.map50), 4)
    map50_95 = round(float(box_metrics.map), 4)

    per_denomination_metrics: Dict[str, Dict[str, float]] = {}
    ap_class_index = getattr(box_metrics, "ap_class_index", [])
    for idx_pos, cls_idx in enumerate(ap_class_index):
        cls_name = CURRENCY_CLASS_NAMES.get(int(cls_idx), f"class_{cls_idx}")
        p_val, r_val, ap50_val, ap_val = box_metrics.class_result(idx_pos)
        per_denomination_metrics[cls_name] = {
            "precision": round(float(p_val), 4),
            "recall": round(float(r_val), 4),
            "mAP50": round(float(ap50_val), 4),
            "mAP50_95": round(float(ap_val), 4),
        }

    cm_graph_path = graphs_dir / "currency_detection_confusion_matrix.png"
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        cm_array = metrics.confusion_matrix.matrix
        labels = [CURRENCY_CLASS_NAMES[i] for i in range(len(CURRENCY_CLASS_NAMES))] + ["Background"]
        if cm_array.shape[0] == len(labels):
            fig, ax = plt.subplots(figsize=(8.5, 7))
            sns.heatmap(
                cm_array.astype(int),
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
            ax.set_title("SafeStep Indian Currency Detection - Test Set Confusion Matrix")
            ax.set_xlabel("True Denomination")
            ax.set_ylabel("Predicted Denomination")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(cm_graph_path, dpi=150)
            plt.close(fig)

    test_img_dir = PROJECT_ROOT / "datasets" / "currency_detection" / "images" / "test"
    inference_times_ms: List[float] = []
    if test_img_dir.exists():
        for img_p in list(test_img_dir.glob("*.jpg"))[:40]:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            t_inf = time.perf_counter()
            model.predict(img, imgsz=img_size, conf=0.25, verbose=False)
            inference_times_ms.append((time.perf_counter() - t_inf) * 1000.0)

    avg_inf_ms = round(float(np.mean(inference_times_ms)), 2) if inference_times_ms else round(float(metrics.speed.get("inference", 25.0)), 2)
    fps = round(1000.0 / max(1.0, avg_inf_ms), 2)

    summary = {
        "model": "SafeStep Indian Currency Detection (YOLOv8n)",
        "weights_path": str(weights_path),
        "evaluated_split": split,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "avg_inference_ms": avg_inf_ms,
        "fps": fps,
        "evaluation_duration_sec": eval_time_s,
        "per_denomination_metrics": per_denomination_metrics,
        "confusion_matrix_graph": str(cm_graph_path),
    }

    out_json = results_dir / "currency_evaluation_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Currency model evaluation complete: P=%.4f, R=%.4f, mAP@50=%.4f, mAP@50-95=%.4f, FPS=%.1f",
        precision,
        recall,
        map50,
        map50_95,
        fps,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SafeStep Indian Currency Detection Model")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    evaluate_currency_model(split=args.split, img_size=args.imgsz)
