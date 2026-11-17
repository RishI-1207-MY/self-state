"""Unit and integration tests for SafeStep Indian Currency Detection Module."""

from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset_generation.generate_currency_dataset import _create_banknote_sprite, _place_rotated_note
from modules.currency_detection import AUTHENTICITY_DISCLAIMER, CurrencyDetectionModule


def test_currency_detection_invalid_input() -> None:
    detector = CurrencyDetectionModule(confidence_threshold=0.25)
    res = detector.detect(np.array([]))
    assert res["detected"] is False
    assert res["currency"] == "INR"
    assert res["authenticity_disclaimer"] == AUTHENTICITY_DISCLAIMER


def test_currency_detection_on_specimen_note() -> None:
    detector = CurrencyDetectionModule(confidence_threshold=0.20)
    test_imgs = list((PROJECT_ROOT / "datasets" / "currency_detection" / "images" / "test").glob("*.jpg"))
    if test_imgs:
        frame = cv2.imread(str(test_imgs[0]))
    else:
        import random
        rng = random.Random(42)
        frame = np.full((416, 416, 3), 120, dtype=np.uint8)
        sprite = _create_banknote_sprite(5, 220, 96, "front", False, rng)
        _place_rotated_note(frame, sprite, 208, 208, 0.0)

    res = detector.detect(frame)
    assert res["currency"] == "INR"
    assert "authenticity_disclaimer" in res
    assert isinstance(res["notes"], list)
    if detector.model_loaded and res["detected"]:
        assert res["denomination"] in (10, 20, 50, 100, 200, 500)
        assert 0.0 < res["confidence"] <= 1.0
