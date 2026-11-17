"""Image preprocessing utilities for SafeStep AI modules."""

from typing import Tuple
import cv2
import numpy as np


def validate_image(image: np.ndarray | None) -> bool:
    """Check whether an image array is valid and non-empty."""
    if image is None or not isinstance(image, np.ndarray):
        return False
    if image.size == 0 or len(image.shape) < 2:
        return False
    return True


def resize_frame(
    frame: np.ndarray, target_size: Tuple[int, int] = (640, 480), keep_aspect_ratio: bool = False
) -> np.ndarray:
    """Resize an input frame to target (width, height)."""
    if not validate_image(frame):
        raise ValueError("Invalid input frame provided to resize_frame.")
    target_w, target_h = target_size
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame

    if not keep_aspect_ratio:
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def enhance_for_ocr(frame: np.ndarray) -> np.ndarray:
    """Enhance contrast and sharpness of environmental text regions for OCR."""
    if not validate_image(frame):
        raise ValueError("Invalid frame for OCR enhancement.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def preprocess_currency_image(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess an image for Indian currency detection with white-balance and contrast normalization."""
    if not validate_image(frame):
        raise ValueError("Invalid frame for currency preprocessing.")
    img = frame.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Apply mild CLAHE on L channel in LAB color space to handle low or uneven lighting
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    balanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return balanced
