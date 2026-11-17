"""Training pipeline for SafeStep Custom Object & Obstacle Detection Model (Dataset A).

Uses pretrained YOLOv8n transfer learning on `datasets/object_detection/data.yaml`.
Saves checkpoints, best weights (`models/object_detection/best.pt`), training loss curves,
and training configuration metadata.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger, load_config

logger = get_logger("SafeStep.Train.Object")


def _generate_training_curves(results_csv: Path, output_graph_path: Path, title: str) -> None:
    """Plot actual training and validation loss + mAP progression from Ultralytics results.csv."""
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Subplot 1: Box, Class, and DFL Losses
    for col, label, color in [
        ("train/box_loss", "Train Box Loss", "#1f77b4"),
        ("val/box_loss", "Val Box Loss", "#ff7f0e"),
        ("train/cls_loss", "Train Cls Loss", "#2ca02c"),
        ("val/cls_loss", "Val Cls Loss", "#d62728"),
    ]:
        if col in df.columns:
            axes[0].plot(epochs, df[col], marker="o", label=label, color=color)
    axes[0].set_title(f"{title} - Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Precision, Recall, mAP@50, mAP@50-95
    for col, label, color in [
        ("metrics/precision(B)", "Precision", "#00b4d8"),
        ("metrics/recall(B)", "Recall", "#7209b7"),
        ("metrics/mAP50(B)", "mAP@50", "#2b9348"),
        ("metrics/mAP50-95(B)", "mAP@50-95", "#f77f00"),
    ]:
        if col in df.columns:
            axes[1].plot(epochs, df[col], marker="s", label=label, color=color)
    axes[1].set_title(f"{title} - Validation Metrics per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_graph_path, dpi=150)
    plt.close(fig)


def train_object_detection_model(
    epochs: int = 5,
    batch_size: int = 16,
    img_size: int = 320,
    learning_rate: float = 0.01,
    device: str = "cpu",
    workers: int = 2,
    fraction: float = 1.0,
) -> Dict[str, Any]:
    """Train custom YOLO object detection model on Dataset A and save artifacts."""
    from ultralytics import YOLO

    data_yaml = PROJECT_ROOT / "datasets" / "object_detection" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")

    pretrained_dir = PROJECT_ROOT / "models" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    base_weights = pretrained_dir / "yolov8n.pt"

    output_model_dir = PROJECT_ROOT / "models" / "object_detection"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_model_dir / "runs"

    logger.info(
        "Starting Object Detection YOLO training: epochs=%d, batch=%d, imgsz=%d, device=%s",
        epochs,
        batch_size,
        img_size,
        device,
    )
    t0 = time.time()
    model = YOLO(str(base_weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,
        device=device,
        workers=workers,
        fraction=fraction,
        project=str(runs_dir),
        name="safestep_object_yolov8n",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    duration_sec = round(time.time() - t0, 2)

    run_folder = runs_dir / "safestep_object_yolov8n"
    best_pt = run_folder / "weights" / "best.pt"
    last_pt = run_folder / "weights" / "last.pt"
    target_best = output_model_dir / "best.pt"
    target_last = output_model_dir / "last.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, target_best)
    elif last_pt.exists():
        shutil.copy2(last_pt, target_best)
    if last_pt.exists():
        shutil.copy2(last_pt, target_last)

    results_csv = run_folder / "results.csv"
    graph_out = PROJECT_ROOT / "evaluation" / "graphs" / "object_detection_training_curves.png"
    _generate_training_curves(results_csv, graph_out, "Object Detection Model")
    if results_csv.exists():
        shutil.copy2(results_csv, PROJECT_ROOT / "evaluation" / "results" / "object_training_results.csv")

    train_meta = {
        "model_name": "safestep_object_yolov8n",
        "base_weights": str(base_weights),
        "best_weights_path": str(target_best),
        "data_yaml": str(data_yaml),
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "learning_rate": learning_rate,
        "device": device,
        "fraction": fraction,
        "training_duration_seconds": duration_sec,
        "training_curves_graph": str(graph_out),
    }
    with open(output_model_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    logger.info("Object detection model training completed in %.1fs -> %s", duration_sec, target_best)
    return train_meta


if __name__ == "__main__":
    cfg = load_config().get("training", {}).get("object_detection", {})
    parser = argparse.ArgumentParser(description="Train SafeStep Object Detection Model")
    parser.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 5)))
    parser.add_argument("--batch", type=int, default=int(cfg.get("batch_size", 16)))
    parser.add_argument("--imgsz", type=int, default=int(cfg.get("img_size", 320)))
    parser.add_argument("--lr", type=float, default=float(cfg.get("learning_rate", 0.01)))
    parser.add_argument("--device", type=str, default=str(cfg.get("device", "cpu")))
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction per epoch")
    args = parser.parse_args()

    train_object_detection_model(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        learning_rate=args.lr,
        device=args.device,
        fraction=args.fraction,
    )
