"""Evaluation script for SafeStep Object Detection Model.

Computes actual Precision, Recall, mAP@50, mAP@50-95, per-class metrics,
confusion matrix heatmap, and environmental condition breakdown (indoor, outdoor,
low_light, bright, crowded_scene, occluded) on the held-out test split.
"""

import argparse
import csv
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

from datasets.dataset_generation.generate_object_dataset import OBJECT_CLASSES
from utils.logger import get_logger

logger = get_logger("SafeStep.Eval.Object")


def evaluate_object_model(
    weights_path: Path | None = None,
    data_yaml: Path | None = None,
    split: str = "test",
    img_size: int = 320,
) -> Dict[str, Any]:
    """Run Ultralytics validation on held-out test set and save real metrics and confusion matrix."""
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = PROJECT_ROOT / "models" / "object_detection" / "best.pt"
    if data_yaml is None:
        data_yaml = PROJECT_ROOT / "datasets" / "object_detection" / "data.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained object detection weights not found at {weights_path}")

    results_dir = PROJECT_ROOT / "evaluation" / "results"
    graphs_dir = PROJECT_ROOT / "evaluation" / "graphs"
    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    for d in (results_dir, graphs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=img_size,
        plots=True,
        project=str(results_dir),
        name="object_eval_run",
        exist_ok=True,
        verbose=False,
    )
    eval_time_s = round(time.perf_counter() - t0, 2)

    box_metrics = metrics.box
    precision = round(float(box_metrics.mp), 4)
    recall = round(float(box_metrics.mr), 4)
    map50 = round(float(box_metrics.map50), 4)
    map50_95 = round(float(box_metrics.map), 4)

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    ap_class_index = getattr(box_metrics, "ap_class_index", [])
    for idx_pos, cls_idx in enumerate(ap_class_index):
        cls_name = OBJECT_CLASSES.get(int(cls_idx), f"class_{cls_idx}")
        p_val, r_val, ap50_val, ap_val = box_metrics.class_result(idx_pos)
        per_class_metrics[cls_name] = {
            "precision": round(float(p_val), 4),
            "recall": round(float(r_val), 4),
            "mAP50": round(float(ap50_val), 4),
            "mAP50_95": round(float(ap_val), 4),
        }

    # Save Confusion Matrix plot using Seaborn
    cm_graph_path = graphs_dir / "object_detection_confusion_matrix.png"
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        cm_array = metrics.confusion_matrix.matrix
        labels = [OBJECT_CLASSES[i] for i in range(len(OBJECT_CLASSES))] + ["Background"]
        if cm_array.shape[0] == len(labels):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_array.astype(int),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
            ax.set_title("SafeStep Object Detection - Test Set Confusion Matrix")
            ax.set_xlabel("True Class")
            ax.set_ylabel("Predicted Class")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(cm_graph_path, dpi=150)
            plt.close(fig)

    # Environmental condition latency & confidence evaluation on actual test images
    meta_path = PROJECT_ROOT / "datasets" / "object_detection" / "metadata.csv"
    test_img_dir = PROJECT_ROOT / "datasets" / "object_detection" / "images" / "test"
    condition_stats: Dict[str, List[float]] = {
        "indoor": [],
        "outdoor": [],
        "low_light": [],
        "bright": [],
        "crowded_scene": [],
        "occluded": [],
    }
    inference_times_ms: List[float] = []

    if meta_path.exists() and test_img_dir.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            test_rows = [
                r for r in csv.DictReader(f)
                if r.get("split") == "test" and str(r.get("is_augmented", "False")).lower() != "true"
            ]
        for row in test_rows[:60]:
            img_p = test_img_dir / row["filename"]
            if not img_p.exists():
                continue
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            t_inf = time.perf_counter()
            preds = model.predict(img, imgsz=img_size, conf=0.25, verbose=False)
            dt_ms = (time.perf_counter() - t_inf) * 1000.0
            inference_times_ms.append(dt_ms)

            confs = []
            for p in preds:
                if p.boxes is not None and len(p.boxes) > 0:
                    confs.extend(p.boxes.conf.cpu().numpy().tolist())
            mean_conf = float(np.mean(confs)) if confs else 0.0

            env = str(row.get("environment", ""))
            light = str(row.get("lighting", ""))
            occ = str(row.get("occluded", "False")).lower() == "true"

            if "indoor" in env:
                condition_stats["indoor"].append(mean_conf)
            if "outdoor" in env:
                condition_stats["outdoor"].append(mean_conf)
            if light in ("low_light", "night"):
                condition_stats["low_light"].append(mean_conf)
            if light in ("bright", "daylight"):
                condition_stats["bright"].append(mean_conf)
            if "crowded" in env:
                condition_stats["crowded_scene"].append(mean_conf)
            if occ:
                condition_stats["occluded"].append(mean_conf)

    avg_inf_ms = round(float(np.mean(inference_times_ms)), 2) if inference_times_ms else round(float(metrics.speed.get("inference", 25.0)), 2)
    fps = round(1000.0 / max(1.0, avg_inf_ms), 2)

    summary = {
        "model": "SafeStep Custom Object Detection (YOLOv8n)",
        "weights_path": str(weights_path),
        "evaluated_split": split,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "avg_inference_ms": avg_inf_ms,
        "fps": fps,
        "evaluation_duration_sec": eval_time_s,
        "per_class_metrics": per_class_metrics,
        "environmental_condition_mean_confidence": {
            k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in condition_stats.items()
        },
        "confusion_matrix_graph": str(cm_graph_path),
    }

    out_json = results_dir / "object_evaluation_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Object model evaluation complete: P=%.4f, R=%.4f, mAP@50=%.4f, mAP@50-95=%.4f, FPS=%.1f",
        precision,
        recall,
        map50,
        map50_95,
        fps,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SafeStep Object Detection Model")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    evaluate_object_model(split=args.split, img_size=args.imgsz)
