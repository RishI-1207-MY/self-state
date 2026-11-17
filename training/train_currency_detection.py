"""Training pipeline for SafeStep Indian Currency Detection Model (Dataset B).

Uses pretrained YOLOv8n transfer learning on `datasets/currency_detection/data.yaml`.
Saves checkpoints, best weights (`models/currency_detection/best.pt`), training loss curves,
and training configuration metadata.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_object_detection import _generate_training_curves
from utils.logger import get_logger, load_config

logger = get_logger("SafeStep.Train.Currency")


def train_currency_detection_model(
    epochs: int = 5,
    batch_size: int = 16,
    img_size: int = 320,
    learning_rate: float = 0.01,
    device: str = "cpu",
    workers: int = 2,
    fraction: float = 1.0,
) -> Dict[str, Any]:
    """Train custom YOLO Indian currency detection model on Dataset B and save artifacts."""
    from ultralytics import YOLO

    data_yaml = PROJECT_ROOT / "datasets" / "currency_detection" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Currency dataset configuration not found: {data_yaml}")

    pretrained_dir = PROJECT_ROOT / "models" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    base_weights = pretrained_dir / "yolov8n.pt"

    output_model_dir = PROJECT_ROOT / "models" / "currency_detection"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_model_dir / "runs"

    logger.info(
        "Starting Indian Currency Detection YOLO training: epochs=%d, batch=%d, imgsz=%d, device=%s",
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
        name="safestep_currency_yolov8n",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    duration_sec = round(time.time() - t0, 2)

    run_folder = runs_dir / "safestep_currency_yolov8n"
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
    graph_out = PROJECT_ROOT / "evaluation" / "graphs" / "currency_detection_training_curves.png"
    _generate_training_curves(results_csv, graph_out, "Indian Currency Detection Model")
    if results_csv.exists():
        shutil.copy2(results_csv, PROJECT_ROOT / "evaluation" / "results" / "currency_training_results.csv")

    train_meta = {
        "model_name": "safestep_currency_yolov8n",
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

    logger.info("Currency detection model training completed in %.1fs -> %s", duration_sec, target_best)
    return train_meta


if __name__ == "__main__":
    cfg = load_config().get("training", {}).get("currency_detection", {})
    parser = argparse.ArgumentParser(description="Train SafeStep Indian Currency Detection Model")
    parser.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 5)))
    parser.add_argument("--batch", type=int, default=int(cfg.get("batch_size", 16)))
    parser.add_argument("--imgsz", type=int, default=int(cfg.get("img_size", 320)))
    parser.add_argument("--lr", type=float, default=float(cfg.get("learning_rate", 0.01)))
    parser.add_argument("--device", type=str, default=str(cfg.get("device", "cpu")))
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction per epoch")
    args = parser.parse_args()

    train_currency_detection_model(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        learning_rate=args.lr,
        device=args.device,
        fraction=args.fraction,
    )
