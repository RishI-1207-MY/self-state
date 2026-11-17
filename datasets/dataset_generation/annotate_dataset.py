"""YOLO-format annotation utility for SafeStep datasets.

Supports:
1. Programmatic / CLI bounding-box conversion from pixel coordinates (x1, y1, x2, y2) to normalized YOLO format.
2. Auto-labeling unannotated real images using a pretrained YOLO model with class mapping.
3. Interactive OpenCV bounding-box drawing tool when run with `--interactive`.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.DatasetGen.Annotate")

# Mapping from standard COCO indices to SafeStep Object Detection classes (0..11)
COCO_TO_SAFESTEP_MAP: Dict[int, int] = {
    0: 0,   # person -> Person (0)
    1: 3,   # bicycle -> Bicycle (3)
    2: 1,   # car -> Car (1)
    3: 2,   # motorcycle -> Motorcycle (2)
    5: 4,   # bus -> Bus (4)
    7: 5,   # truck -> Truck (5)
    56: 8,  # chair -> Chair (8)
    60: 9,  # dining table -> Table (9)
}


def pixel_box_to_yolo(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert pixel (x1, y1, x2, y2) to normalized YOLO (x_center, y_center, width, height)."""
    xa, xb = sorted((max(0, min(img_w, x1)), max(0, min(img_w, x2))))
    ya, yb = sorted((max(0, min(img_h, y1)), max(0, min(img_h, y2))))
    bw = max(1, xb - xa)
    bh = max(1, yb - ya)
    cx = (xa + bw / 2.0) / img_w
    cy = (ya + bh / 2.0) / img_h
    return round(cx, 6), round(cy, 6), round(bw / img_w, 6), round(bh / img_h, 6)


def write_yolo_annotation(
    label_path: Path, boxes: List[Tuple[int, int, int, int, int]], img_w: int, img_h: int
) -> None:
    """Write a list of (class_id, x1, y1, x2, y2) pixel boxes to a YOLO .txt file."""
    lines: List[str] = []
    for cls_id, x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = pixel_box_to_yolo(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"{cls_id} {cx} {cy} {bw} {bh}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def auto_annotate_unlabeled_images(dataset_dir: Path, weights_path: Path, conf: float = 0.35) -> int:
    """Auto-annotate any images in raw/images that have empty label files using YOLO weights."""
    from ultralytics import YOLO

    raw_img_dir = dataset_dir / "raw" / "images"
    raw_lbl_dir = dataset_dir / "raw" / "labels"
    if not raw_img_dir.exists() or not weights_path.exists():
        return 0

    model = YOLO(str(weights_path))
    updated = 0
    for img_path in raw_img_dir.glob("*.jpg"):
        lbl_path = raw_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        results = model.predict(img, conf=conf, verbose=False)
        boxes: List[Tuple[int, int, int, int, int]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                coco_cls = int(box.cls[0].item())
                mapped_cls = COCO_TO_SAFESTEP_MAP.get(coco_cls, 11)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append((mapped_cls, x1, y1, x2, y2))
        if boxes:
            write_yolo_annotation(lbl_path, boxes, w, h)
            updated += 1
    logger.info("Auto-annotated %d unlabeled images in %s", updated, dataset_dir)
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeStep YOLO Annotation Utility")
    parser.add_argument("--dataset-dir", type=str, default=str(PROJECT_ROOT / "datasets" / "object_detection"))
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "models" / "pretrained" / "yolov8n.pt"))
    args = parser.parse_args()
    auto_annotate_unlabeled_images(Path(args.dataset_dir), Path(args.weights))
