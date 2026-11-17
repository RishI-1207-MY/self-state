"""Programmatic and webcam/real-assisted generator for Dataset A (Object and Obstacle Detection).

Generates diverse indoor and outdoor navigation scenes across 12 classes:
0: Person, 1: Car, 2: Motorcycle, 3: Bicycle, 4: Bus, 5: Truck,
6: Pole, 7: Wall, 8: Chair, 9: Table, 10: Door, 11: Other obstacles.

Clearly distinguishes synthetic scenes from real/webcam captures in metadata.csv.
"""

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger("SafeStep.DatasetGen.Object")

OBJECT_CLASSES: Dict[int, str] = {
    0: "Person",
    1: "Car",
    2: "Motorcycle",
    3: "Bicycle",
    4: "Bus",
    5: "Truck",
    6: "Pole",
    7: "Wall",
    8: "Chair",
    9: "Table",
    10: "Door",
    11: "Other obstacles",
}

ENVIRONMENTS = [
    ("corridor", "indoor"),
    ("room", "indoor"),
    ("office", "indoor"),
    ("road", "outdoor"),
    ("footpath", "outdoor"),
    ("public_space", "outdoor"),
    ("crowded_scene", "outdoor"),
]

LIGHTING_CONDITIONS = ["daylight", "bright", "low_light", "night", "indoor_warm"]
CAMERA_ANGLES = ["eye_level", "slightly_tilted_down", "wide_angle", "left_oblique", "right_oblique"]


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return max(0, min(w - 1, x1)), max(0, min(h - 1, y1)), max(1, min(w, x2)), max(1, min(h, y2))


def _to_yolo_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    cx = (x1 + bw / 2.0) / w
    cy = (y1 + bh / 2.0) / h
    return (
        round(min(max(cx, 0.001), 0.999), 6),
        round(min(max(cy, 0.001), 0.999), 6),
        round(min(max(bw / w, 0.005), 0.999), 6),
        round(min(max(bh / h, 0.005), 0.999), 6),
    )


def _render_background(
    w: int, h: int, env_name: str, env_type: str, rng: random.Random
) -> np.ndarray:
    """Render an indoor or outdoor structured background with horizon, perspective lines, and texture."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    horizon_y = rng.randint(int(h * 0.32), int(h * 0.52))

    if env_type == "indoor":
        ceil_col = (rng.randint(185, 235), rng.randint(190, 235), rng.randint(195, 240))
        wall_col = (rng.randint(140, 210), rng.randint(145, 210), rng.randint(150, 215))
        floor_col = (rng.randint(80, 145), rng.randint(85, 145), rng.randint(90, 155))
        canvas[: int(horizon_y * 0.45), :] = ceil_col
        canvas[int(horizon_y * 0.45) : horizon_y, :] = wall_col
        canvas[horizon_y:, :] = floor_col
        # Perspective corridor/room lines
        vp_x = rng.randint(int(w * 0.35), int(w * 0.65))
        cv2.line(canvas, (vp_x, horizon_y), (0, h), (60, 60, 65), 2)
        cv2.line(canvas, (vp_x, horizon_y), (w, h), (60, 60, 65), 2)
        # Floor tiles
        for ty in range(horizon_y + 20, h, 35):
            cv2.line(canvas, (0, ty), (w, ty), (70, 75, 80), 1)
    else:
        sky_col = (rng.randint(180, 245), rng.randint(150, 215), rng.randint(100, 175))
        ground_col = (rng.randint(65, 115), rng.randint(70, 115), rng.randint(70, 115))
        canvas[:horizon_y, :] = sky_col
        canvas[horizon_y:, :] = ground_col
        # Road/footpath markings
        vp_x = rng.randint(int(w * 0.35), int(w * 0.65))
        road_pts = np.array([[vp_x - 45, horizon_y], [vp_x + 45, horizon_y], [w - 20, h], [20, h]], np.int32)
        cv2.fillPoly(canvas, [road_pts], (55, 58, 62))
        for dash_y in range(horizon_y + 15, h - 15, 40):
            cv2.line(canvas, (vp_x, dash_y), (vp_x, min(h - 5, dash_y + 20)), (210, 220, 230), 3)

    # Add subtle natural sensor grain
    noise = np.random.default_rng(rng.randint(0, 100000)).normal(0, 5.5, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def _draw_object_instance(
    canvas: np.ndarray,
    cls_id: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    rng: random.Random,
) -> None:
    """Draw a visually distinctive, structured object instance with class-specific visual features."""
    bw = max(12, x2 - x1)
    bh = max(12, y2 - y1)

    if cls_id == 0:  # Person: head, torso, legs, clothing colors
        head_r = max(4, int(min(bw, bh) * 0.16))
        cx = x1 + bw // 2
        head_cy = y1 + head_r + 2
        skin = (rng.randint(110, 180), rng.randint(135, 200), rng.randint(165, 235))
        shirt = (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220))
        pants = (rng.randint(25, 110), rng.randint(30, 110), rng.randint(40, 130))
        cv2.circle(canvas, (cx, head_cy), head_r, skin, -1)
        torso_y1 = head_cy + head_r
        torso_y2 = y1 + int(bh * 0.62)
        cv2.rectangle(canvas, (x1 + int(bw * 0.2), torso_y1), (x2 - int(bw * 0.2), torso_y2), shirt, -1)
        cv2.rectangle(canvas, (x1 + int(bw * 0.22), torso_y2), (cx - 2, y2), pants, -1)
        cv2.rectangle(canvas, (cx + 2, torso_y2), (x2 - int(bw * 0.22), y2), pants, -1)

    elif cls_id in (1, 4, 5):  # Car (1), Bus (4), Truck (5)
        body_col = {
            1: (rng.randint(40, 200), rng.randint(40, 180), rng.randint(60, 230)),
            4: (rng.randint(20, 90), rng.randint(140, 220), rng.randint(200, 250)),
            5: (rng.randint(60, 140), rng.randint(80, 150), rng.randint(120, 200)),
        }[cls_id]
        roof_y = y1 + int(bh * (0.28 if cls_id == 1 else 0.12))
        cv2.rectangle(canvas, (x1, roof_y), (x2, y2 - int(bh * 0.12)), body_col, -1)
        cv2.rectangle(canvas, (x1 + int(bw * 0.12), y1), (x2 - int(bw * 0.12), roof_y + 4), body_col, -1)
        # Windshield / windows
        win_col = (210, 225, 235)
        if cls_id == 4:  # Bus multi-window row
            step = max(8, bw // 5)
            for wx in range(x1 + 6, x2 - step, step):
                cv2.rectangle(canvas, (wx, y1 + int(bh * 0.16)), (wx + step - 4, y1 + int(bh * 0.45)), win_col, -1)
        else:
            cv2.rectangle(
                canvas,
                (x1 + int(bw * 0.18), y1 + int(bh * 0.08)),
                (x2 - int(bw * 0.18), roof_y + int(bh * 0.15)),
                win_col,
                -1,
            )
        # Wheels and headlights
        wheel_r = max(4, int(min(bw, bh) * 0.14))
        cv2.circle(canvas, (x1 + int(bw * 0.24), y2 - wheel_r), wheel_r, (25, 25, 25), -1)
        cv2.circle(canvas, (x2 - int(bw * 0.24), y2 - wheel_r), wheel_r, (25, 25, 25), -1)
        cv2.circle(canvas, (x1 + 6, y2 - int(bh * 0.32)), max(2, wheel_r // 2), (180, 245, 255), -1)
        cv2.circle(canvas, (x2 - 6, y2 - int(bh * 0.32)), max(2, wheel_r // 2), (180, 245, 255), -1)

    elif cls_id in (2, 3):  # Motorcycle (2) or Bicycle (3)
        wheel_r = max(5, int(min(bw, bh) * 0.22))
        w1_x, w2_x = x1 + wheel_r + 2, x2 - wheel_r - 2
        wy = y2 - wheel_r - 1
        thickness = 3 if cls_id == 2 else 2
        cv2.circle(canvas, (w1_x, wy), wheel_r, (30, 30, 30), thickness)
        cv2.circle(canvas, (w2_x, wy), wheel_r, (30, 30, 30), thickness)
        frame_col = (20, 60, 200) if cls_id == 2 else (200, 120, 20)
        mid_x = (w1_x + w2_x) // 2
        seat_y = y1 + int(bh * 0.35)
        cv2.line(canvas, (w1_x, wy), (mid_x, seat_y), frame_col, thickness + 1)
        cv2.line(canvas, (w2_x, wy), (mid_x, seat_y), frame_col, thickness + 1)
        cv2.line(canvas, (w2_x, wy), (x2 - int(bw * 0.15), y1 + 4), frame_col, thickness)
        if cls_id == 2:
            cv2.rectangle(canvas, (mid_x - bw // 5, seat_y - 4), (mid_x + bw // 5, wy - 4), frame_col, -1)

    elif cls_id == 6:  # Pole
        pole_col = (135, 140, 145)
        px1 = x1 + int(bw * 0.25)
        px2 = x2 - int(bw * 0.25)
        cv2.rectangle(canvas, (px1, y1), (px2, y2), pole_col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + max(6, int(bh * 0.08))), (90, 95, 100), -1)
        # Yellow hazard stripes near base
        stripe_y = y1 + int(bh * 0.65)
        cv2.rectangle(canvas, (px1, stripe_y), (px2, min(y2, stripe_y + int(bh * 0.15))), (0, 210, 245), -1)

    elif cls_id == 7:  # Wall
        wall_col = (rng.randint(120, 175), rng.randint(115, 165), rng.randint(135, 190))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), wall_col, -1)
        step_y = max(8, bh // 6)
        step_x = max(12, bw // 5)
        for by in range(y1, y2, step_y):
            cv2.line(canvas, (x1, by), (x2, by), (95, 95, 105), 1)
            offset = (step_x // 2) if ((by - y1) // step_y) % 2 == 1 else 0
            for bx in range(x1 + offset, x2, step_x):
                cv2.line(canvas, (bx, by), (bx, min(y2, by + step_y)), (95, 95, 105), 1)

    elif cls_id == 8:  # Chair
        chair_col = (rng.randint(50, 130), rng.randint(70, 150), rng.randint(120, 200))
        seat_y = y1 + int(bh * 0.52)
        # Backrest
        cv2.rectangle(canvas, (x1 + int(bw * 0.15), y1), (x2 - int(bw * 0.15), seat_y), chair_col, -1)
        # Seat
        cv2.rectangle(canvas, (x1 + int(bw * 0.08), seat_y), (x2 - int(bw * 0.08), seat_y + max(5, int(bh * 0.1))), chair_col, -1)
        # Legs
        cv2.line(canvas, (x1 + int(bw * 0.15), seat_y), (x1 + int(bw * 0.15), y2), (50, 50, 60), 3)
        cv2.line(canvas, (x2 - int(bw * 0.15), seat_y), (x2 - int(bw * 0.15), y2), (50, 50, 60), 3)

    elif cls_id == 9:  # Table
        tbl_col = (rng.randint(60, 115), rng.randint(95, 155), rng.randint(140, 210))
        top_h = max(6, int(bh * 0.18))
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + top_h), tbl_col, -1)
        cv2.line(canvas, (x1 + int(bw * 0.12), y1 + top_h), (x1 + int(bw * 0.12), y2), tbl_col, 4)
        cv2.line(canvas, (x2 - int(bw * 0.12), y1 + top_h), (x2 - int(bw * 0.12), y2), tbl_col, 4)

    elif cls_id == 10:  # Door
        door_col = (rng.randint(65, 120), rng.randint(95, 150), rng.randint(140, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 225), 3)
        cv2.rectangle(canvas, (x1 + 3, y1 + 3), (x2 - 3, y2 - 2), door_col, -1)
        # Door panels and handle
        cv2.rectangle(canvas, (x1 + int(bw * 0.18), y1 + int(bh * 0.1)), (x2 - int(bw * 0.18), y1 + int(bh * 0.45)), (55, 80, 125), 2)
        cv2.rectangle(canvas, (x1 + int(bw * 0.18), y1 + int(bh * 0.55)), (x2 - int(bw * 0.18), y2 - int(bh * 0.1)), (55, 80, 125), 2)
        cv2.circle(canvas, (x2 - int(bw * 0.18), (y1 + y2) // 2), max(3, bw // 14), (40, 215, 245), -1)

    else:  # 11: Other obstacles (traffic cone / barrier / bollard / box)
        cone_pts = np.array(
            [[(x1 + x2) // 2, y1], [x2, y2], [x1, y2]], dtype=np.int32
        )
        cv2.fillPoly(canvas, [cone_pts], (0, 110, 255))
        band_y1 = y1 + int(bh * 0.38)
        band_y2 = y1 + int(bh * 0.56)
        cv2.rectangle(canvas, (x1 + int(bw * 0.25), band_y1), (x2 - int(bw * 0.25), band_y2), (240, 240, 245), -1)
        cv2.rectangle(canvas, (x1, y2 - max(4, int(bh * 0.1))), (x2, y2), (40, 40, 45), -1)


def _apply_lighting_and_occlusion(
    canvas: np.ndarray, lighting: str, occluded: bool, rng: random.Random
) -> np.ndarray:
    img = canvas.copy()
    if lighting == "low_light":
        img = cv2.convertScaleAbs(img, alpha=0.62, beta=-15)
    elif lighting == "night":
        img = cv2.convertScaleAbs(img, alpha=0.45, beta=-25)
    elif lighting == "bright":
        img = cv2.convertScaleAbs(img, alpha=1.18, beta=18)
    elif lighting == "indoor_warm":
        b, g, r = cv2.split(img)
        r = cv2.add(r, 15)
        b = cv2.subtract(b, 10)
        img = cv2.merge((b, g, r))

    if occluded:
        h, w = img.shape[:2]
        occ_w = rng.randint(int(w * 0.05), int(w * 0.12))
        occ_x = rng.randint(int(w * 0.15), int(w * 0.80))
        cv2.rectangle(img, (occ_x, int(h * 0.45)), (occ_x + occ_w, h), (45, 48, 52), -1)

    return img


def generate_object_dataset(
    output_root: Path,
    num_base_images: int = 750,
    img_size: Tuple[int, int] = (416, 416),
    seed: int = 42,
    try_webcam_capture: bool = True,
) -> Dict[str, Any]:
    """Generate raw base object detection dataset with YOLO annotations and metadata records."""
    rng = random.Random(seed)
    raw_img_dir = output_root / "raw" / "images"
    raw_lbl_dir = output_root / "raw" / "labels"
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_lbl_dir.mkdir(parents=True, exist_ok=True)

    w, h = img_size
    metadata_rows: List[Dict[str, Any]] = []
    webcam_captured = 0

    # Attempt to capture a few genuine camera frames if a local webcam is attached
    if try_webcam_capture:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            for idx in range(min(5, num_base_images)):
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break
                resized = cv2.resize(frame, (w, h))
                # Overlay a synthetic calibration obstacle box on real webcam background so annotation is exact
                cls_id = idx % len(OBJECT_CLASSES)
                x1, y1, x2, y2 = int(w * 0.32), int(h * 0.35), int(w * 0.68), int(h * 0.82)
                _draw_object_instance(resized, cls_id, x1, y1, x2, y2, rng)
                img_id = f"obj_webcam_{idx:05d}"
                img_path = raw_img_dir / f"{img_id}.jpg"
                lbl_path = raw_lbl_dir / f"{img_id}.txt"
                cv2.imwrite(str(img_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cx, cy, bw, bh = _to_yolo_bbox(x1, y1, x2, y2, w, h)
                lbl_path.write_text(f"{cls_id} {cx} {cy} {bw} {bh}\n", encoding="utf-8")
                metadata_rows.append(
                    {
                        "image_id": img_id,
                        "filename": f"{img_id}.jpg",
                        "source_type": "webcam_hybrid_scene",
                        "is_synthetic": True,
                        "session_id": f"session_webcam_{idx // 2:03d}",
                        "environment": "indoor_webcam",
                        "lighting": "indoor_warm",
                        "camera_angle": "eye_level",
                        "occluded": False,
                        "num_objects": 1,
                        "classes_present": str(cls_id),
                        "split": "unassigned",
                    }
                )
                webcam_captured += 1
            cap.release()

    remaining = num_base_images - webcam_captured
    num_classes = len(OBJECT_CLASSES)

    for i in range(remaining):
        env_name, env_type = rng.choice(ENVIRONMENTS)
        lighting = rng.choice(LIGHTING_CONDITIONS)
        angle = rng.choice(CAMERA_ANGLES)
        occluded = rng.random() < 0.25
        session_id = f"session_obj_{i // 10:04d}"

        canvas = _render_background(w, h, env_name, env_type, rng)
        num_objs = rng.randint(2, 4)
        yolo_lines: List[str] = []
        classes_in_img: List[int] = []

        # Ensure balanced representation across all 12 classes
        primary_cls = (i + rng.randint(0, 2)) % num_classes
        chosen_classes = [primary_cls] + [rng.randint(0, num_classes - 1) for _ in range(num_objs - 1)]

        # Place objects in non-colliding horizontal slots with depth variation
        slot_w = w // num_objs
        for slot_idx, cls_id in enumerate(chosen_classes):
            scale = rng.uniform(0.18, 0.42)
            aspect = {
                0: rng.uniform(0.42, 0.62),  # Person
                6: rng.uniform(0.22, 0.35),  # Pole
                10: rng.uniform(0.48, 0.65), # Door
            }.get(cls_id, rng.uniform(0.75, 1.35))

            obj_h = int(h * scale)
            obj_w = max(20, min(slot_w - 10, int(obj_h * aspect)))
            slot_x1 = slot_idx * slot_w + 5
            x1 = rng.randint(slot_x1, max(slot_x1, (slot_idx + 1) * slot_w - obj_w - 5))
            y1 = rng.randint(int(h * 0.28), max(int(h * 0.29), h - obj_h - 10))
            x2, y2 = _clip_box(x1, y1, x1 + obj_w, y1 + obj_h, w, h)[2:]

            _draw_object_instance(canvas, cls_id, x1, y1, x2, y2, rng)
            cx, cy, bw, bh = _to_yolo_bbox(x1, y1, x2, y2, w, h)
            yolo_lines.append(f"{cls_id} {cx} {cy} {bw} {bh}")
            classes_in_img.append(cls_id)

        canvas = _apply_lighting_and_occlusion(canvas, lighting, occluded, rng)
        img_id = f"obj_synth_{i:05d}"
        img_path = raw_img_dir / f"{img_id}.jpg"
        lbl_path = raw_lbl_dir / f"{img_id}.txt"

        cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        metadata_rows.append(
            {
                "image_id": img_id,
                "filename": f"{img_id}.jpg",
                "source_type": "programmatic_synthetic",
                "is_synthetic": True,
                "session_id": session_id,
                "environment": f"{env_name}_{env_type}",
                "lighting": lighting,
                "camera_angle": angle,
                "occluded": occluded,
                "num_objects": len(classes_in_img),
                "classes_present": ";".join(str(c) for c in classes_in_img),
                "split": "unassigned",
            }
        )

    raw_meta_path = output_root / "raw_metadata.csv"
    if metadata_rows:
        with open(raw_meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)

    logger.info(
        "Generated %d base object detection images (%d webcam-assisted, %d programmatic synthetic) at %s",
        len(metadata_rows),
        webcam_captured,
        remaining,
        output_root,
    )
    return {
        "total_base_images": len(metadata_rows),
        "webcam_hybrid_images": webcam_captured,
        "synthetic_images": remaining,
        "raw_metadata_path": str(raw_meta_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SafeStep Object & Obstacle Detection base dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "object_detection"),
        help="Target dataset directory",
    )
    parser.add_argument("--count", type=int, default=750, help="Number of base images to generate before augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate_object_dataset(Path(args.output_dir), num_base_images=args.count, seed=args.seed)
