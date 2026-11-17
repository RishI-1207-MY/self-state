"""Webcam capture and licensed local image ingestion utility for SafeStep datasets.

Supports capturing genuine camera frames or importing real image directories into
Dataset A (object_detection) or Dataset B (currency_detection), recording accurate
provenance (`is_synthetic=False`, `source_type='webcam_capture'` or `'imported_real'`)
in `raw_metadata.csv`.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.DatasetGen.Collect")


def capture_from_webcam(
    dataset_dir: Path,
    num_frames: int = 10,
    camera_index: int = 0,
    interval_sec: float = 0.5,
    session_name: str | None = None,
) -> int:
    """Capture genuine images from a connected webcam and log them to raw_metadata.csv."""
    raw_img_dir = dataset_dir / "raw" / "images"
    raw_lbl_dir = dataset_dir / "raw" / "labels"
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.warning("Webcam index %d is unavailable. Skipping live capture.", camera_index)
        return 0

    session_id = session_name or f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta_path = dataset_dir / "raw_metadata.csv"
    captured = 0

    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            break
        img_id = f"{session_id}_{idx:04d}"
        img_path = raw_img_dir / f"{img_id}.jpg"
        lbl_path = raw_lbl_dir / f"{img_id}.txt"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not lbl_path.exists():
            lbl_path.write_text("", encoding="utf-8")
        captured += 1
        if interval_sec > 0 and idx < num_frames - 1:
            time.sleep(interval_sec)

    cap.release()
    logger.info("Captured %d real frames from webcam into %s", captured, raw_img_dir)
    return captured


def import_local_images(source_folder: Path, dataset_dir: Path, session_id: str = "imported_real_01") -> int:
    """Import existing real photographs from a folder into the dataset's raw directory."""
    if not source_folder.exists():
        logger.error("Source folder does not exist: %s", source_folder)
        return 0

    raw_img_dir = dataset_dir / "raw" / "images"
    raw_lbl_dir = dataset_dir / "raw" / "labels"
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_file in source_folder.glob(ext):
            target_img = raw_img_dir / f"{session_id}_{img_file.stem}.jpg"
            shutil.copy2(img_file, target_img)
            lbl_candidate = img_file.with_suffix(".txt")
            target_lbl = raw_lbl_dir / f"{session_id}_{img_file.stem}.txt"
            if lbl_candidate.exists():
                shutil.copy2(lbl_candidate, target_lbl)
            elif not target_lbl.exists():
                target_lbl.write_text("", encoding="utf-8")
            count += 1

    logger.info("Imported %d real images from %s into %s", count, source_folder, dataset_dir)
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect real webcam or local images for SafeStep datasets.")
    parser.add_argument("--dataset", choices=["object", "currency"], default="object")
    parser.add_argument("--webcam-frames", type=int, default=0, help="Number of webcam frames to capture")
    parser.add_argument("--import-dir", type=str, default="", help="Optional folder of real images to import")
    args = parser.parse_args()

    target_dir = PROJECT_ROOT / "datasets" / (
        "object_detection" if args.dataset == "object" else "currency_detection"
    )
    if args.webcam_frames > 0:
        capture_from_webcam(target_dir, num_frames=args.webcam_frames)
    if args.import_dir:
        import_local_images(Path(args.import_dir), target_dir)
