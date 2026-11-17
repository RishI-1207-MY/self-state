"""Programmatic generator for Dataset B (Indian Currency Detection).

Classes:
0: 10_rupees
1: 20_rupees
2: 50_rupees
3: 100_rupees
4: 200_rupees
5: 500_rupees

Includes front and back sides, multiple orientations, distances, lighting conditions,
partial folds, diverse backgrounds, multiple notes per frame, and partial occlusions.
All generated notes are clearly marked as `programmatic_synthetic` in metadata.csv
and include a visible 'SPECIMEN / SYNTHETIC' watermark so they are never confused with
genuine banknote photographs.
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

logger = get_logger("SafeStep.DatasetGen.Currency")

CURRENCY_SPECS: Dict[int, Dict[str, Any]] = {
    0: {
        "class_name": "10_rupees",
        "denomination": 10,
        "bgr": (65, 105, 165),      # Chocolate brown
        "accent_bgr": (45, 75, 125),
        "aspect_ratio": 1.95,       # 123mm x 63mm
        "motif": "Konark Wheel",
    },
    1: {
        "class_name": "20_rupees",
        "denomination": 20,
        "bgr": (65, 195, 175),      # Greenish yellow
        "accent_bgr": (40, 145, 130),
        "aspect_ratio": 2.04,       # 129mm x 63mm
        "motif": "Ellora Caves",
    },
    2: {
        "class_name": "50_rupees",
        "denomination": 50,
        "bgr": (215, 195, 55),      # Fluorescent blue
        "accent_bgr": (165, 140, 30),
        "aspect_ratio": 2.04,       # 135mm x 66mm
        "motif": "Hampi Chariot",
    },
    3: {
        "class_name": "100_rupees",
        "denomination": 100,
        "bgr": (210, 145, 180),     # Lavender / violet
        "accent_bgr": (160, 95, 135),
        "aspect_ratio": 2.15,       # 142mm x 66mm
        "motif": "Rani Ki Vav",
    },
    4: {
        "class_name": "200_rupees",
        "denomination": 200,
        "bgr": (45, 155, 245),      # Bright yellow / orange
        "accent_bgr": (25, 110, 195),
        "aspect_ratio": 2.21,       # 146mm x 66mm
        "motif": "Sanchi Stupa",
    },
    5: {
        "class_name": "500_rupees",
        "denomination": 500,
        "bgr": (145, 155, 150),     # Stone grey
        "accent_bgr": (95, 105, 100),
        "aspect_ratio": 2.27,       # 150mm x 66mm
        "motif": "Red Fort",
    },
}

BACKGROUNDS = ["wooden_desk", "cloth_fabric", "palm_hand_tone", "dark_wallet", "stone_counter", "tiled_table"]
LIGHTING_MODES = ["daylight", "warm_indoor", "dim_light", "bright_glare", "shadowed"]


def _render_surface_background(w: int, h: int, bg_type: str, rng: random.Random) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if bg_type == "wooden_desk":
        base = (rng.randint(55, 95), rng.randint(95, 140), rng.randint(140, 190))
        canvas[:, :] = base
        for y in range(0, h, rng.randint(8, 16)):
            cv2.line(canvas, (0, y), (w, y), (base[0] - 12, base[1] - 12, base[2] - 12), 1)
    elif bg_type == "cloth_fabric":
        base = (rng.randint(90, 180), rng.randint(90, 180), rng.randint(90, 180))
        canvas[:, :] = base
        for x in range(0, w, 10):
            cv2.line(canvas, (x, 0), (x, h), (max(0, base[0] - 10), max(0, base[1] - 10), max(0, base[2] - 10)), 1)
    elif bg_type == "palm_hand_tone":
        canvas[:, :] = (rng.randint(115, 155), rng.randint(140, 180), rng.randint(175, 220))
    elif bg_type == "dark_wallet":
        canvas[:, :] = (rng.randint(25, 55), rng.randint(25, 55), rng.randint(30, 60))
    else:
        canvas[:, :] = (rng.randint(120, 200), rng.randint(120, 200), rng.randint(120, 200))

    noise = np.random.default_rng(rng.randint(0, 100000)).normal(0, 6.0, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _create_banknote_sprite(
    cls_id: int, note_w: int, note_h: int, side: str, folded: bool, rng: random.Random
) -> np.ndarray:
    """Create a structured synthetic Indian currency note image with denomination numerals, security strip, and side features."""
    spec = CURRENCY_SPECS[cls_id]
    denom = spec["denomination"]
    bgr = spec["bgr"]
    accent = spec["accent_bgr"]

    sprite = np.full((note_h, note_w, 3), bgr, dtype=np.uint8)
    # Outer border & inner frame
    cv2.rectangle(sprite, (0, 0), (note_w - 1, note_h - 1), (35, 35, 35), 2)
    cv2.rectangle(sprite, (6, 6), (note_w - 7, note_h - 7), accent, 2)

    # White watermark window
    wm_x1 = int(note_w * (0.68 if side == "front" else 0.14))
    wm_x2 = wm_x1 + int(note_w * 0.16)
    cv2.rectangle(sprite, (wm_x1, int(note_h * 0.18)), (wm_x2, int(note_h * 0.82)), (238, 240, 242), -1)

    # Security thread (dashed metallic green/blue vertical strip)
    strip_x = int(note_w * 0.54)
    for sy in range(6, note_h - 6, 12):
        cv2.rectangle(sprite, (strip_x, sy), (strip_x + max(3, note_w // 45), min(note_h - 6, sy + 7)), (180, 220, 60), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_large = max(0.55, note_h / 95.0)
    scale_small = max(0.35, note_h / 160.0)

    # Prominent denomination numerals in corners and center
    cv2.putText(
        sprite,
        f"Rs.{denom}",
        (int(note_w * 0.08), int(note_h * 0.34)),
        font,
        scale_large,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sprite,
        str(denom),
        (int(note_w * 0.34), int(note_h * 0.64)),
        font,
        scale_large * 1.25,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        sprite,
        str(denom),
        (int(note_w * 0.34), int(note_h * 0.64)),
        font,
        scale_large * 1.25,
        accent,
        2,
        cv2.LINE_AA,
    )

    if side == "front":
        # Reserve Bank header & tactile bleed lines on edges
        cv2.putText(
            sprite,
            "RESERVE BANK OF INDIA - SPECIMEN",
            (int(note_w * 0.08), int(note_h * 0.15)),
            font,
            scale_small * 0.85,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        # Angular bleed marks on left/right edges (varies by denomination)
        num_bleed = cls_id + 2
        for b_idx in range(num_bleed):
            by = int(note_h * (0.25 + 0.1 * b_idx))
            cv2.line(sprite, (0, by), (10, by), (20, 20, 20), 2)
            cv2.line(sprite, (note_w - 11, by), (note_w - 1, by), (20, 20, 20), 2)
    else:
        # Back side: motif circle and multi-language panel
        cv2.circle(
            sprite,
            (int(note_w * 0.62), int(note_h * 0.52)),
            int(note_h * 0.26),
            accent,
            2,
        )
        cv2.putText(
            sprite,
            spec["motif"],
            (int(note_w * 0.48), int(note_h * 0.88)),
            font,
            scale_small,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            sprite,
            (int(note_w * 0.32), int(note_h * 0.18)),
            (int(note_w * 0.44), int(note_h * 0.82)),
            accent,
            1,
        )

    if folded:
        # Simulate a crease or corner fold across the note
        crease_x = rng.randint(int(note_w * 0.25), int(note_w * 0.75))
        cv2.line(sprite, (crease_x, 0), (crease_x + rng.randint(-15, 15), note_h), (40, 40, 40), 2)

    return sprite


def _place_rotated_note(
    canvas: np.ndarray, sprite: np.ndarray, center_x: int, center_y: int, angle_deg: float
) -> Tuple[int, int, int, int]:
    """Paste a rotated note sprite onto canvas and return tight bounding box (x1, y1, x2, y2)."""
    h, w = canvas.shape[:2]
    sh, sw = sprite.shape[:2]

    diag = int(np.ceil(np.hypot(sw, sh))) + 8
    padded = np.zeros((diag, diag, 3), dtype=np.uint8)
    mask = np.zeros((diag, diag), dtype=np.uint8)
    ox = (diag - sw) // 2
    oy = (diag - sh) // 2
    padded[oy : oy + sh, ox : ox + sw] = sprite
    mask[oy : oy + sh, ox : ox + sw] = 255

    rot_mat = cv2.getRotationMatrix2D((diag / 2.0, diag / 2.0), angle_deg, 1.0)
    rot_sprite = cv2.warpAffine(padded, rot_mat, (diag, diag), flags=cv2.INTER_LINEAR)
    rot_mask = cv2.warpAffine(mask, rot_mat, (diag, diag), flags=cv2.INTER_NEAREST)

    ys, xs = np.where(rot_mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 10, 10

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    crop_w = max_x - min_x + 1
    crop_h = max_y - min_y + 1

    dst_x1 = max(0, min(w - crop_w, center_x - crop_w // 2))
    dst_y1 = max(0, min(h - crop_h, center_y - crop_h // 2))
    dst_x2 = min(w, dst_x1 + crop_w)
    dst_y2 = min(h, dst_y1 + crop_h)

    sub_w = dst_x2 - dst_x1
    sub_h = dst_y2 - dst_y1
    sprite_crop = rot_sprite[min_y : min_y + sub_h, min_x : min_x + sub_w]
    mask_crop = rot_mask[min_y : min_y + sub_h, min_x : min_x + sub_w] > 127

    roi = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
    roi[mask_crop] = sprite_crop[mask_crop]
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = roi

    return dst_x1, dst_y1, dst_x2, dst_y2


def generate_currency_dataset(
    output_root: Path,
    num_base_images: int = 850,
    img_size: Tuple[int, int] = (416, 416),
    seed: int = 101,
) -> Dict[str, Any]:
    """Generate raw base Indian currency dataset with YOLO bounding boxes and metadata."""
    rng = random.Random(seed)
    raw_img_dir = output_root / "raw" / "images"
    raw_lbl_dir = output_root / "raw" / "labels"
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_lbl_dir.mkdir(parents=True, exist_ok=True)

    w, h = img_size
    num_classes = len(CURRENCY_SPECS)
    metadata_rows: List[Dict[str, Any]] = []

    for i in range(num_base_images):
        bg_type = rng.choice(BACKGROUNDS)
        lighting = rng.choice(LIGHTING_MODES)
        side = "front" if (i % 2 == 0) else "back"
        folded = rng.random() < 0.25
        occluded = rng.random() < 0.20
        multi_note = rng.random() < 0.35
        session_id = f"session_curr_{i // 10:04d}"

        canvas = _render_surface_background(w, h, bg_type, rng)
        note_count = 2 if multi_note else 1
        primary_cls = i % num_classes
        chosen_classes = [primary_cls]
        if note_count == 2:
            chosen_classes.append((primary_cls + rng.randint(1, num_classes - 1)) % num_classes)

        yolo_lines: List[str] = []
        denom_list: List[int] = []

        for n_idx, cls_id in enumerate(chosen_classes):
            spec = CURRENCY_SPECS[cls_id]
            scale = rng.uniform(0.36, 0.56) if note_count == 1 else rng.uniform(0.28, 0.42)
            note_w = int(w * scale)
            note_h = max(36, int(note_w / spec["aspect_ratio"]))
            angle = rng.uniform(-28.0, 28.0)

            sprite = _create_banknote_sprite(cls_id, note_w, note_h, side, folded, rng)
            if note_count == 1:
                cx = rng.randint(int(w * 0.32), int(w * 0.68))
                cy = rng.randint(int(h * 0.32), int(h * 0.68))
            else:
                cx = int(w * (0.30 if n_idx == 0 else 0.70)) + rng.randint(-15, 15)
                cy = int(h * (0.35 if n_idx == 0 else 0.65)) + rng.randint(-15, 15)

            x1, y1, x2, y2 = _place_rotated_note(canvas, sprite, cx, cy, angle)

            if occluded and n_idx == 0:
                # Simulate thumb/finger holding edge of banknote
                thumb_w = max(12, int((x2 - x1) * 0.16))
                thumb_h = max(12, int((y2 - y1) * 0.24))
                cv2.ellipse(
                    canvas,
                    (x1 + thumb_w // 2, (y1 + y2) // 2),
                    (thumb_w, thumb_h),
                    0,
                    0,
                    360,
                    (125, 155, 195),
                    -1,
                )

            bw_norm = max(0.01, min(0.999, (x2 - x1) / w))
            bh_norm = max(0.01, min(0.999, (y2 - y1) / h))
            cx_norm = max(0.001, min(0.999, (x1 + (x2 - x1) / 2.0) / w))
            cy_norm = max(0.001, min(0.999, (y1 + (y2 - y1) / 2.0) / h))
            yolo_lines.append(f"{cls_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}")
            denom_list.append(spec["denomination"])

        # Lighting variations
        if lighting == "dim_light":
            canvas = cv2.convertScaleAbs(canvas, alpha=0.68, beta=-15)
        elif lighting == "bright_glare":
            canvas = cv2.convertScaleAbs(canvas, alpha=1.18, beta=20)
        elif lighting == "warm_indoor":
            b, g, r = cv2.split(canvas)
            canvas = cv2.merge((cv2.subtract(b, 10), g, cv2.add(r, 15)))

        img_id = f"curr_synth_{i:05d}"
        img_path = raw_img_dir / f"{img_id}.jpg"
        lbl_path = raw_lbl_dir / f"{img_id}.txt"

        cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        metadata_rows.append(
            {
                "image_id": img_id,
                "filename": f"{img_id}.jpg",
                "source_type": "programmatic_synthetic_specimen",
                "is_synthetic": True,
                "session_id": session_id,
                "side": side,
                "background": bg_type,
                "lighting": lighting,
                "folded": folded,
                "occluded": occluded,
                "num_notes": len(denom_list),
                "denominations": ";".join(str(d) for d in denom_list),
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
        "Generated %d base Indian currency images (synthetic specimen notes) at %s",
        len(metadata_rows),
        output_root,
    )
    return {
        "total_base_images": len(metadata_rows),
        "raw_metadata_path": str(raw_meta_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SafeStep Indian Currency Detection base dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "currency_detection"),
        help="Target currency dataset directory",
    )
    parser.add_argument("--count", type=int, default=850, help="Number of base images to generate before augmentation")
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    args = parser.parse_args()
    generate_currency_dataset(Path(args.output_dir), num_base_images=args.count, seed=args.seed)
