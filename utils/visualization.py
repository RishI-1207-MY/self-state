"""Visualization utilities for bounding boxes, OCR overlays, HUD banners, and dataset charts."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

# High-contrast accessibility color palette (BGR)
COLOR_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 180),    # High-contrast spring green
    (0, 215, 255),    # Bright gold/yellow
    (255, 180, 0),    # Bright cyan-blue
    (147, 20, 255),   # Deep magenta
    (0, 140, 255),    # Bright orange
    (50, 205, 50),    # Lime green
    (255, 255, 0),    # Aqua
    (203, 192, 255),  # Pink
    (0, 69, 255),     # Red-orange
    (255, 144, 30),   # Dodger blue
    (128, 255, 0),    # Chartreuse
    (0, 0, 255),      # Alert red
]


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    fps: float | None = None,
    distance_m: float | None = None,
    sensor_simulated: bool = True,
    latest_alert: str | None = None,
) -> np.ndarray:
    """Draw accessible, high-contrast bounding boxes and HUD telemetry on a frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cls_id = int(det.get("class_id", 0))
        cls_name = str(det.get("class_name", "Object"))
        conf = float(det.get("confidence", 0.0))
        is_hazard = bool(det.get("is_hazard", False))

        color = (0, 60, 255) if is_hazard else COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} {conf * 100:.1f}%"
        if det.get("position"):
            label += f" [{det['position']}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - th - 10)
        bg_y2 = max(th + 10, y1)
        cv2.rectangle(canvas, (x1, bg_y1), (min(w, x1 + tw + 10), bg_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 5, bg_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Top status bar
    cv2.rectangle(canvas, (0, 0), (w, 36), (20, 20, 20), -1)
    hud_items = [f"Objects: {len(detections)}"]
    if fps is not None:
        hud_items.append(f"FPS: {fps:.1f}")
    if distance_m is not None:
        mode_tag = "SIM" if sensor_simulated else "HW"
        hud_items.append(f"Dist ({mode_tag}): {distance_m:.2f}m")
    hud_text = " | ".join(hud_items)
    cv2.putText(
        canvas,
        hud_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )

    if latest_alert:
        cv2.rectangle(canvas, (0, h - 38), (w, h), (15, 15, 35), -1)
        cv2.putText(
            canvas,
            f"Voice: {latest_alert[:75]}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_ocr_results(frame: np.ndarray, ocr_items: List[Dict[str, Any]]) -> np.ndarray:
    """Draw OCR bounding polygons/boxes and recognized text on a frame."""
    canvas = frame.copy()
    for item in ocr_items:
        bbox = item.get("bbox")
        text = str(item.get("text", ""))
        conf = float(item.get("confidence", 0.0))
        if bbox is not None:
            pts = np.array(bbox, dtype=np.int32)
            if pts.shape == (4, 2):
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                x, y = int(pts[0][0]), max(20, int(pts[0][1]) - 8)
            elif len(pts) == 4:
                x1, y1, x2, y2 = [int(v) for v in pts]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
                x, y = x1, max(20, y1 - 8)
            else:
                x, y = 10, 30
            label = f"{text} ({conf * 100:.0f}%)"
            cv2.putText(
                canvas,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return canvas


def draw_currency_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw currency bounding boxes and denomination badges on a frame."""
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
        denom = det.get("denomination", "Unknown")
        conf = float(det.get("confidence", 0.0))
        label = f"INR Rs.{denom} ({conf * 100:.1f}%)"
        color = (0, 215, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 12)), (x1 + tw + 12, max(th + 12, y1)), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 6, max(th + 6, y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas
