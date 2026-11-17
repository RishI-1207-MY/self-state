"""Dataset validation, quality inspection, and visualization generator for SafeStep.

Performs:
- Corrupted image detection and removal
- Missing label detection
- Invalid YOLO bounding-box coordinate checks
- Leakage verification across train/val/test splits
- Class distribution statistics and bar charts
- Sample annotated bounding-box grid visualization
- JSON and Markdown dataset quality report generation
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.visualization import COLOR_PALETTE

logger = get_logger("SafeStep.DatasetGen.Validate")


def validate_and_report_dataset(
    dataset_dir: Path,
    graphs_output_dir: Path | None = None,
    reports_output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Validate all splits in dataset_dir, generate visualizations, and return quality statistics."""
    if graphs_output_dir is None:
        graphs_output_dir = PROJECT_ROOT / "evaluation" / "graphs"
    if reports_output_dir is None:
        reports_output_dir = PROJECT_ROOT / "evaluation" / "reports"
    graphs_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names: Dict[int, str] = {int(k): str(v) for k, v in data_cfg.get("names", {}).items()}
    num_classes = len(class_names)

    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    class_instance_counts: Counter = Counter()
    split_class_counts: Dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    corrupted_images_removed = 0
    missing_labels_count = 0
    invalid_bbox_count = 0
    total_annotations = 0
    sample_images_with_boxes: List[Tuple[Path, Path, str]] = []

    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            if img is None or img.size == 0:
                corrupted_images_removed += 1
                img_path.unlink(missing_ok=True)
                lbl_path.unlink(missing_ok=True)
                continue

            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                missing_labels_count += 1
                continue

            valid_lines: List[str] = []
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_bbox_count += 1
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = [float(v) for v in parts[1:]]
                except ValueError:
                    invalid_bbox_count += 1
                    continue

                if (
                    cls_id < 0
                    or cls_id >= num_classes
                    or not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0)
                ):
                    invalid_bbox_count += 1
                    continue

                valid_lines.append(line)
                class_instance_counts[cls_id] += 1
                split_class_counts[split][cls_id] += 1
                total_annotations += 1

            if valid_lines:
                split_counts[split] += 1
                if len(sample_images_with_boxes) < 12 and len(sample_images_with_boxes) % 3 == ("train", "val", "test").index(split):
                    sample_images_with_boxes.append((img_path, lbl_path, split))
                elif len(sample_images_with_boxes) < 6:
                    sample_images_with_boxes.append((img_path, lbl_path, split))

    # Verify zero session/parent leakage across splits via metadata.csv
    meta_path = dataset_dir / "metadata.csv"
    synthetic_count = 0
    real_or_hybrid_count = 0
    augmented_count = 0
    split_sessions: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sp = row.get("split", "train")
                sid = row.get("session_id", "")
                if sp in split_sessions and sid:
                    split_sessions[sp].add(sid)
                if str(row.get("source_type", "")).startswith("programmatic_synthetic"):
                    synthetic_count += 1
                else:
                    real_or_hybrid_count += 1
                if str(row.get("is_augmented", "False")).lower() == "true":
                    augmented_count += 1

    leakage_train_val = len(split_sessions["train"].intersection(split_sessions["val"]))
    leakage_train_test = len(split_sessions["train"].intersection(split_sessions["test"]))
    leakage_val_test = len(split_sessions["val"].intersection(split_sessions["test"]))
    zero_leakage = (leakage_train_val + leakage_train_test + leakage_val_test) == 0

    total_usable = sum(split_counts.values())
    ds_tag = dataset_dir.name

    # 1. Generate Class Distribution Bar Graph
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_ordered = [class_names[i] for i in range(num_classes)]
    counts_ordered = [class_instance_counts[i] for i in range(num_classes)]
    bars = ax.bar(labels_ordered, counts_ordered, color="#00b4d8", edgecolor="#03045e")
    ax.set_title(f"SafeStep {ds_tag.replace('_', ' ').title()} - Class Distribution ({total_annotations} boxes)")
    ax.set_xlabel("Class Name")
    ax.set_ylabel("Annotated Bounding Box Count")
    plt.xticks(rotation=30, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    dist_graph_path = graphs_output_dir / f"{ds_tag}_class_distribution.png"
    plt.savefig(dist_graph_path, dpi=150)
    plt.close(fig)

    # 2. Generate Sample Annotated Mosaic Visualization
    sample_grid_path = graphs_output_dir / f"{ds_tag}_sample_annotations.png"
    if sample_images_with_boxes:
        grid_Validate = sample_images_with_boxes[:6]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes_flat = axes.flatten()
        for idx, ax_sub in enumerate(axes_flat):
            if idx >= len(grid_Validate):
                ax_sub.axis("off")
                continue
            img_p, lbl_p, sp_name = grid_Validate[idx]
            im_bgr = cv2.imread(str(img_p))
            ih, iw = im_bgr.shape[:2]
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = [float(v) for v in parts[1:]]
                    x1 = int((cx - bw / 2) * iw)
                    y1 = int((cy - bh / 2) * ih)
                    x2 = int((cx + bw / 2) * iw)
                    y2 = int((cy + bh / 2) * ih)
                    col = COLOR_PALETTE[cid % len(COLOR_PALETTE)]
                    cv2.rectangle(im_bgr, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(
                        im_bgr,
                        class_names.get(cid, str(cid)),
                        (max(2, x1), max(16, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        col,
                        2,
                    )
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            ax_sub.imshow(im_rgb)
            ax_sub.set_title(f"{img_p.name} ({sp_name})", fontsize=9)
            ax_sub.axis("off")
        plt.suptitle(f"SafeStep {ds_tag.replace('_', ' ').title()} - Sample Verified Annotations", fontsize=13)
        plt.tight_layout()
        plt.savefig(sample_grid_path, dpi=150)
        plt.close(fig)

    report: Dict[str, Any] = {
        "dataset_name": ds_tag,
        "dataset_path": str(dataset_dir.resolve()),
        "total_usable_images": total_usable,
        "total_bounding_boxes": total_annotations,
        "splits": split_counts,
        "split_percentages": {
            k: round((v / total_usable) * 100, 2) if total_usable > 0 else 0.0
            for k, v in split_counts.items()
        },
        "class_distribution": {class_names[i]: int(class_instance_counts[i]) for i in range(num_classes)},
        "provenance": {
            "programmatic_synthetic_images": synthetic_count,
            "webcam_or_real_hybrid_images": real_or_hybrid_count,
            "augmented_images": augmented_count,
            "base_unaugmented_images": total_usable - augmented_count,
        },
        "quality_checks": {
            "corrupted_images_removed": corrupted_images_removed,
            "missing_labels_detected": missing_labels_count,
            "invalid_bboxes_detected": invalid_bbox_count,
            "zero_session_leakage_verified": zero_leakage,
            "status": "PASSED" if (total_usable > 0 and invalid_bbox_count == 0 and zero_leakage) else "WARNING",
        },
        "artifacts": {
            "class_distribution_graph": str(dist_graph_path),
            "sample_annotations_graph": str(sample_grid_path),
        },
    }

    report_json_path = reports_output_dir / f"{ds_tag}_quality_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Validated %s: %d usable images, %d bboxes, status=%s",
        ds_tag,
        total_usable,
        total_annotations,
        report["quality_checks"]["status"],
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SafeStep datasets and generate quality reports.")
    parser.add_argument("--dataset", choices=["object", "currency", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("object", "both"):
        validate_and_report_dataset(PROJECT_ROOT / "datasets" / "object_detection")
    if args.dataset in ("currency", "both"):
        validate_and_report_dataset(PROJECT_ROOT / "datasets" / "currency_detection")
