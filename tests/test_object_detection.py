"""Unit and integration tests for SafeStep Object Detection Module."""

from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.object_detection import ObjectDetectionModule, SimpleDetectionTracker


def test_tracker_persistence_and_iou() -> None:
    tracker = SimpleDetectionTracker(max_disappeared=5, iou_threshold=0.3)
    frame1_dets = [
        {"class_id": 0, "class_name": "Person", "confidence": 0.91, "bbox": [100, 100, 200, 300], "is_hazard": True}
    ]
    tracked1 = tracker.update(frame1_dets)
    assert len(tracked1) == 1
    assert tracked1[0]["is_new_track"] is True
    tid = tracked1[0]["track_id"]

    # Slightly shifted box in frame 2 should preserve track_id and mark is_new_track=False
    frame2_dets = [
        {"class_id": 0, "class_name": "Person", "confidence": 0.89, "bbox": [105, 102, 205, 302], "is_hazard": True}
    ]
    tracked2 = tracker.update(frame2_dets)
    assert tracked2[0]["track_id"] == tid
    assert tracked2[0]["is_new_track"] is False


def test_object_detector_invalid_and_valid_frames() -> None:
    detector = ObjectDetectionModule(confidence_threshold=0.25)
    # Invalid frame handling
    bad_res = detector.detect(None)  # type: ignore[arg-type]
    assert bad_res["num_objects"] == 0
    assert "error" in bad_res

    # Valid test image from Dataset A if available, else synthetic canvas
    test_imgs = list((PROJECT_ROOT / "datasets" / "object_detection" / "images" / "test").glob("*.jpg"))
    if test_imgs:
        frame = cv2.imread(str(test_imgs[0]))
    else:
        frame = np.full((416, 416, 3), 140, dtype=np.uint8)
        cv2.rectangle(frame, (120, 100), (260, 360), (40, 90, 200), -1)

    res = detector.detect(frame)
    assert "detections" in res
    assert "inference_ms" in res
    assert res["inference_ms"] >= 0.0
    assert res["fps"] >= 0.0
