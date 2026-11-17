"""Indian Currency Detection & Denomination Recognition Module for SafeStep.

Classes:
0: 10_rupees   -> 10
1: 20_rupees   -> 20
2: 50_rupees   -> 50
3: 100_rupees  -> 100
4: 200_rupees  -> 200
5: 500_rupees  -> 500

Returns structured output matching the research specification:
{
  "detected": true,
  "currency": "INR",
  "denomination": 500,
  "confidence": 0.94,
  "notes": [...],
  "total_value_inr": 500,
  "authenticity_disclaimer": "..."
}

IMPORTANT: Visual denomination recognition classifies only the apparent denomination
of Indian banknotes and does NOT verify legal authenticity or counterfeit security features.
"""

from pathlib import Path
import time
from typing import Any, Dict, List
import cv2
import numpy as np

from utils.logger import get_logger, get_project_root
from utils.preprocessing import preprocess_currency_image, validate_image

logger = get_logger("SafeStep.CurrencyDetection")

CURRENCY_CLASS_TO_DENOM: Dict[int, int] = {
    0: 10,
    1: 20,
    2: 50,
    3: 100,
    4: 200,
    5: 500,
}

CURRENCY_CLASS_NAMES: Dict[int, str] = {
    0: "10_rupees",
    1: "20_rupees",
    2: "50_rupees",
    3: "100_rupees",
    4: "200_rupees",
    5: "500_rupees",
}

SPOKEN_RUPEES: Dict[int, str] = {
    10: "Ten rupees",
    20: "Twenty rupees",
    50: "Fifty rupees",
    100: "One hundred rupees",
    200: "Two hundred rupees",
    500: "Five hundred rupees",
}

AUTHENTICITY_DISCLAIMER = (
    "Visual denomination recognition only classifies the apparent denomination of Indian banknotes "
    "and does NOT verify whether a note is genuine or authentic."
)


class CurrencyDetectionModule:
    """Custom YOLO model inference module for Indian currency denomination recognition."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        img_size: int = 416,
    ) -> None:
        self.root = get_project_root()
        self.weights_path = (
            Path(weights_path)
            if weights_path
            else self.root / "models" / "currency_detection" / "best.pt"
        )
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.img_size = int(img_size)

        self.model: Any = None
        self.model_loaded = False
        self.last_inference_ms = 0.0
        self.load_model()

    def load_model(self) -> bool:
        """Load the custom-trained Indian currency YOLO model."""
        try:
            from ultralytics import YOLO

            if self.weights_path.exists():
                self.model = YOLO(str(self.weights_path))
                self.model_loaded = True
                logger.info("Loaded custom Indian currency detection model: %s", self.weights_path)
                return True
            logger.warning("Custom currency model not yet found at %s.", self.weights_path)
            self.model_loaded = False
            return False
        except Exception as exc:
            logger.error("Failed to load currency model: %s", exc)
            self.model_loaded = False
            return False

    def detect(
        self, frame: np.ndarray, confidence_threshold: float | None = None
    ) -> Dict[str, Any]:
        """Detect one or more Indian currency notes in `frame` and return structured JSON-ready output."""
        if not validate_image(frame):
            return {
                "detected": False,
                "currency": "INR",
                "denomination": None,
                "confidence": 0.0,
                "notes": [],
                "total_value_inr": 0,
                "inference_ms": 0.0,
                "authenticity_disclaimer": AUTHENTICITY_DISCLAIMER,
                "error": "Invalid input image",
            }

        t0 = time.perf_counter()
        if not self.model_loaded:
            self.load_model()

        conf_thresh = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        notes: List[Dict[str, Any]] = []
        if self.model is not None:
            proc_frame = preprocess_currency_image(frame)
            results = self.model.predict(
                proc_frame,
                conf=conf_thresh,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
            )
            # If preprocessed frame yielded no detections, also check original frame
            if all(r.boxes is None or len(r.boxes) == 0 for r in results):
                results = self.model.predict(
                    frame,
                    conf=conf_thresh,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    verbose=False,
                )

            for res in results:
                if res.boxes is None:
                    continue
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = round(float(box.conf[0].item()), 4)
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    denom = CURRENCY_CLASS_TO_DENOM.get(cls_id, 100)
                    cls_name = CURRENCY_CLASS_NAMES.get(cls_id, f"{denom}_rupees")
                    notes.append(
                        {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "currency": "INR",
                            "denomination": denom,
                            "spoken_denomination": SPOKEN_RUPEES.get(denom, f"{denom} rupees"),
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

        notes.sort(key=lambda x: x["confidence"], reverse=True)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.last_inference_ms = elapsed_ms

        if notes:
            top = notes[0]
            total_val = sum(int(n["denomination"]) for n in notes)
            spoken_list = [n["spoken_denomination"] for n in notes]
            announcement = (
                f"Indian currency detected. {', '.join(spoken_list)}."
                if len(notes) == 1
                else f"Indian currency detected. {len(notes)} notes: {', '.join(spoken_list)}. Total {total_val} rupees."
            )
            return {
                "detected": True,
                "currency": "INR",
                "denomination": top["denomination"],
                "confidence": top["confidence"],
                "notes": notes,
                "total_value_inr": total_val,
                "announcement": announcement,
                "inference_ms": elapsed_ms,
                "authenticity_disclaimer": AUTHENTICITY_DISCLAIMER,
            }

        return {
            "detected": False,
            "currency": "INR",
            "denomination": None,
            "confidence": 0.0,
            "notes": [],
            "total_value_inr": 0,
            "announcement": "",
            "inference_ms": elapsed_ms,
            "authenticity_disclaimer": AUTHENTICITY_DISCLAIMER,
        }
