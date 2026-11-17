"""Leakage-free dataset splitting utility for SafeStep datasets.

Splits raw base images into train (70%), val (20%), and test (10%) splits
grouped by `session_id` / original source image BEFORE any augmentation is performed,
guaranteeing zero data leakage between training, validation, and testing sets.
Also writes `data.yaml` and `metadata.csv` for each dataset.
"""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.DatasetGen.Split")


def split_dataset_by_session(
    dataset_dir: Path,
    class_names: Dict[int, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, int]:
    """Split raw/images and raw/labels into train/val/test by session_id without leakage."""
    raw_img_dir = dataset_dir / "raw" / "images"
    raw_lbl_dir = dataset_dir / "raw" / "labels"
    raw_meta_path = dataset_dir / "raw_metadata.csv"

    if not raw_img_dir.exists() or not raw_meta_path.exists():
        raise FileNotFoundError(f"Raw dataset or metadata missing in {dataset_dir}")

    # Clean existing split folders to prevent stale files
    for split in ("train", "val", "test"):
        img_split_dir = dataset_dir / "images" / split
        lbl_split_dir = dataset_dir / "labels" / split
        if img_split_dir.exists():
            shutil.rmtree(img_split_dir)
        if lbl_split_dir.exists():
            shutil.rmtree(lbl_split_dir)
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group rows by session_id so all frames from a session go into the same split
    sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = row.get("session_id") or row["image_id"]
        sessions[sid].append(row)

    session_ids = sorted(sessions.keys())
    rng = random.Random(seed)
    rng.shuffle(session_ids)

    n_sessions = len(session_ids)
    n_train = int(round(n_sessions * train_ratio))
    n_val = int(round(n_sessions * val_ratio))

    train_sids = set(session_ids[:n_train])
    val_sids = set(session_ids[n_train : n_train + n_val])

    updated_rows: List[Dict[str, Any]] = []
    counts = {"train": 0, "val": 0, "test": 0}

    for sid, group in sessions.items():
        if sid in train_sids:
            split = "train"
        elif sid in val_sids:
            split = "val"
        else:
            split = "test"

        for row in group:
            img_name = row["filename"]
            img_id = row["image_id"]
            src_img = raw_img_dir / img_name
            src_lbl = raw_lbl_dir / f"{img_id}.txt"
            if not src_img.exists() or not src_lbl.exists():
                continue

            dst_img = dataset_dir / "images" / split / img_name
            dst_lbl = dataset_dir / "labels" / split / f"{img_id}.txt"
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

            row_copy = dict(row)
            row_copy["split"] = split
            row_copy["is_augmented"] = False
            row_copy["parent_image_id"] = img_id
            updated_rows.append(row_copy)
            counts[split] += 1

    meta_out = dataset_dir / "metadata.csv"
    if updated_rows:
        with open(meta_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(updated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(updated_rows)

    # Write Ultralytics data.yaml with absolute path
    data_yaml_content = {
        "path": str(dataset_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {int(k): str(v) for k, v in class_names.items()},
    }
    with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)

    logger.info(
        "Split dataset at %s by session_id -> train=%d, val=%d, test=%d (total=%d)",
        dataset_dir.name,
        counts["train"],
        counts["val"],
        counts["test"],
        sum(counts.values()),
    )
    return counts


if __name__ == "__main__":
    from datasets.dataset_generation.generate_currency_dataset import CURRENCY_SPECS
    from datasets.dataset_generation.generate_object_dataset import OBJECT_CLASSES

    parser = argparse.ArgumentParser(description="Split SafeStep datasets by session_id before augmentation.")
    parser.add_argument("--dataset", choices=["object", "currency", "both"], default="both")
    args = parser.parse_args()

    if args.dataset in ("object", "both"):
        split_dataset_by_session(PROJECT_ROOT / "datasets" / "object_detection", OBJECT_CLASSES)
    if args.dataset in ("currency", "both"):
        curr_classes = {k: v["class_name"] for k, v in CURRENCY_SPECS.items()}
        split_dataset_by_session(PROJECT_ROOT / "datasets" / "currency_detection", curr_classes)
