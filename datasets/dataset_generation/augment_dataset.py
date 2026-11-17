"""Dataset augmentation pipeline using Albumentations with YOLO bounding-box preservation.

Applies realistic photometric and geometric transformations AFTER split_dataset.py
so augmented children remain strictly inside the same split as their parent image
(zero train/val/test leakage) and never silently duplicate identical files.
"""

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple
import albumentations as A
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.DatasetGen.Augment")


def _build_augmentation_pipeline(allow_flip: bool = True) -> A.Compose:
    transforms = [
        A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.8),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=18, p=0.6),
        A.GaussNoise(p=0.4),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.Affine(scale=(0.88, 1.12), translate_percent=(-0.06, 0.06), rotate=(-12, 12), p=0.75),
    ]
    if allow_flip:
        transforms.insert(0, A.HorizontalFlip(p=0.5))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.35),
    )


def _read_yolo_labels(lbl_path: Path) -> Tuple[List[List[float]], List[int]]:
    bboxes: List[List[float]] = []
    labels: List[int] = []
    if not lbl_path.exists():
        return bboxes, labels

    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(x) for x in parts[1:]]
        # Clamp strictly within (0, 1] for Albumentations
        bw = max(0.005, min(0.99, bw))
        bh = max(0.005, min(0.99, bh))
        cx = max(bw / 2.0 + 0.001, min(1.0 - bw / 2.0 - 0.001, cx))
        cy = max(bh / 2.0 + 0.001, min(1.0 - bh / 2.0 - 0.001, cy))
        bboxes.append([cx, cy, bw, bh])
        labels.append(cls_id)
    return bboxes, labels


def augment_split_dataset(
    dataset_dir: Path,
    augmentations_per_image: Dict[str, int] | None = None,
    allow_horizontal_flip: bool = True,
    seed: int = 42,
) -> Dict[str, int]:
    """Augment images within each split and update metadata.csv."""
    if augmentations_per_image is None:
        augmentations_per_image = {"train": 2, "val": 2, "test": 2}

    random.seed(seed)
    np.random.seed(seed)
    pipeline = _build_augmentation_pipeline(allow_flip=allow_horizontal_flip)

    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}. Run split_dataset.py first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        base_rows = [r for r in csv.DictReader(f) if str(r.get("is_augmented", "False")).lower() != "true"]

    new_rows: List[Dict[str, Any]] = list(base_rows)
    aug_counts = {"train": 0, "val": 0, "test": 0}

    for row in base_rows:
        split = row["split"]
        k = augmentations_per_image.get(split, 0)
        if k <= 0:
            continue

        img_id = row["image_id"]
        img_path = dataset_dir / "images" / split / row["filename"]
        lbl_path = dataset_dir / "labels" / split / f"{img_id}.txt"
        if not img_path.exists() or not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        bboxes, class_labels = _read_yolo_labels(lbl_path)
        if not bboxes:
            continue

        for aug_idx in range(1, k + 1):
            try:
                transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                t_img = transformed["image"]
                t_boxes = transformed["bboxes"]
                t_labels = transformed["class_labels"]
                if not t_boxes:
                    continue

                aug_id = f"{img_id}_aug{aug_idx}"
                aug_filename = f"{aug_id}.jpg"
                dst_img = dataset_dir / "images" / split / aug_filename
                dst_lbl = dataset_dir / "labels" / split / f"{aug_id}.txt"

                cv2.imwrite(str(dst_img), t_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                yolo_lines = [
                    f"{int(cls)} {bx[0]:.6f} {bx[1]:.6f} {bx[2]:.6f} {bx[3]:.6f}"
                    for cls, bx in zip(t_labels, t_boxes)
                ]
                dst_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

                aug_row = dict(row)
                aug_row["image_id"] = aug_id
                aug_row["filename"] = aug_filename
                aug_row["is_augmented"] = True
                aug_row["parent_image_id"] = img_id
                new_rows.append(aug_row)
                aug_counts[split] += 1
            except Exception as exc:
                logger.debug("Skipped single augmentation for %s: %s", img_id, exc)

    if new_rows:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
            writer.writeheader()
            writer.writerows(new_rows)

    logger.info(
        "Augmented dataset %s -> added train=%d, val=%d, test=%d (total images now=%d)",
        dataset_dir.name,
        aug_counts["train"],
        aug_counts["val"],
        aug_counts["test"],
        len(new_rows),
    )
    return {"total_images": len(new_rows), **aug_counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment SafeStep datasets with Albumentations.")
    parser.add_argument("--dataset", choices=["object", "currency", "both"], default="both")
    parser.add_argument("--multiplier", type=int, default=2, help="Number of augmented variants per base image")
    args = parser.parse_args()

    mult = {"train": args.multiplier, "val": args.multiplier, "test": args.multiplier}
    if args.dataset in ("object", "both"):
        augment_split_dataset(PROJECT_ROOT / "datasets" / "object_detection", mult, allow_horizontal_flip=True)
    if args.dataset in ("currency", "both"):
        augment_split_dataset(PROJECT_ROOT / "datasets" / "currency_detection", mult, allow_horizontal_flip=False)
