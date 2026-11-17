"""Optical Character Recognition (OCR) Module for SafeStep using EasyOCR.

Features:
- Extracts environmental text from signboards, room numbers, notices, and labels
- Contrast enhancement preprocessing for low-light or blurred scenes
- Confidence filtering and temporal duplicate text suppression
- Manual single-shot trigger and configurable interval automatic OCR mode
- Converts recognized text into spoken announcements via AudioFeedbackEngine
"""

import time
from typing import Any, Dict, List
import cv2
import numpy as np

from utils.logger import get_logger
from utils.preprocessing import enhance_for_ocr, validate_image

logger = get_logger("SafeStep.OCR")


class OCRModule:
    """EasyOCR wrapper with confidence filtering, deduplication, and audio integration."""

    def __init__(
        self,
        languages: List[str] | None = None,
        gpu: bool = False,
        confidence_threshold: float = 0.40,
        min_text_length: int = 2,
        deduplication_window_seconds: float = 10.0,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = bool(gpu)
        self.confidence_threshold = float(confidence_threshold)
        self.min_text_length = int(min_text_length)
        self.deduplication_window_seconds = float(deduplication_window_seconds)

        self.reader: Any = None
        self.engine_ready = False
        self._recent_texts: Dict[str, float] = {}
        self.last_inference_ms: float = 0.0

    def initialize_reader(self) -> bool:
        """Lazy-load EasyOCR Reader so startup stays responsive."""
        if self.engine_ready and self.reader is not None:
            return True
        try:
            import easyocr
            from utils.logger import get_project_root

            local_ocr_dir = get_project_root() / "models" / "easyocr"
            local_ocr_dir.mkdir(parents=True, exist_ok=True)
            if (local_ocr_dir / "craft_mlt_25k.pth").exists():
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    model_storage_directory=str(local_ocr_dir),
                    download_enabled=False,
                    verbose=False,
                )
            else:
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=self.gpu,
                    model_storage_directory=str(local_ocr_dir),
                    verbose=False,
                )
            self.engine_ready = True
            logger.info("Initialized EasyOCR Reader for languages: %s", self.languages)
            return True
        except Exception as exc:
            logger.error("EasyOCR initialization failed: %s", exc)
            self.engine_ready = False
            return False

    def extract_text(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
        deduplicate: bool = False,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Run OCR on an image frame and return filtered text items and combined reading string."""
        if not validate_image(frame):
            return {
                "detected": False,
                "combined_text": "",
                "announcement": "",
                "items": [],
                "inference_ms": 0.0,
                "error": "Invalid input image",
            }

        t0 = time.perf_counter()
        if not self.engine_ready:
            self.initialize_reader()

        conf_thresh = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        items: List[Dict[str, Any]] = []
        seen_in_frame = set()
        now = time.time()

        if self.reader is not None:
            try:
                proc_img = enhance_for_ocr(frame) if preprocess else frame
                raw_results = self.reader.readtext(proc_img)
                # Also fallback to raw un-enhanced frame if enhanced found nothing
                if not raw_results and preprocess:
                    raw_results = self.reader.readtext(frame)

                for bbox, text, conf in raw_results:
                    clean = " ".join(str(text).strip().split())
                    if len(clean) < self.min_text_length or float(conf) < conf_thresh:
                        continue
                    norm_key = clean.lower()
                    if norm_key in seen_in_frame:
                        continue
                    seen_in_frame.add(norm_key)

                    if deduplicate:
                        last_seen = self._recent_texts.get(norm_key, 0.0)
                        if (now - last_seen) < self.deduplication_window_seconds:
                            continue
                    self._recent_texts[norm_key] = now

                    items.append(
                        {
                            "text": clean,
                            "confidence": round(float(conf), 4),
                            "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
                        }
                    )
            except Exception as exc:
                logger.error("Error during EasyOCR inference: %s", exc)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.last_inference_ms = elapsed_ms

        combined = " ".join(item["text"] for item in items)
        announcement = f"Text detected. Reading: {combined}" if combined else ""

        return {
            "detected": len(items) > 0,
            "combined_text": combined,
            "announcement": announcement,
            "items": items,
            "inference_ms": elapsed_ms,
        }
