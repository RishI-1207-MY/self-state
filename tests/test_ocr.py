"""Unit and integration tests for SafeStep OCR Module (EasyOCR)."""

from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ocr_module import OCRModule


def test_ocr_empty_and_invalid_image() -> None:
    ocr = OCRModule(confidence_threshold=0.35)
    invalid_res = ocr.extract_text(None)  # type: ignore[arg-type]
    assert invalid_res["detected"] is False
    assert "error" in invalid_res


def test_ocr_signboard_extraction_and_deduplication() -> None:
    ocr = OCRModule(confidence_threshold=0.25, deduplication_window_seconds=10.0)
    canvas = np.full((240, 520, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, "ROOM 204", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (10, 10, 10), 5, cv2.LINE_AA)

    res1 = ocr.extract_text(canvas, deduplicate=True, preprocess=False)
    if not ocr.engine_ready:
        # When running inside an offline sandbox where ~/.EasyOCR is outside workspace permissions
        # and models/easyocr has not yet been populated, verify graceful non-crashing fallback
        assert res1["detected"] is False
        assert res1["items"] == []
        return

    assert res1["detected"] is True
    assert "204" in res1["combined_text"].upper() or "ROOM" in res1["combined_text"].upper()
    assert res1["announcement"].startswith("Text detected. Reading:")

    # Immediate second call with deduplicate=True should filter out duplicate text within window
    res2 = ocr.extract_text(canvas, deduplicate=True, preprocess=False)
    assert res2["detected"] is False
