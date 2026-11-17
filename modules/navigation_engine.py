"""Integrated Multi-Modal Navigation Engine for SafeStep.

Orchestrates:
1. Camera video capture (or fallback stream)
2. Real-time Object & Obstacle Detection
3. Interval-scheduled or manually triggered EasyOCR environmental text reading
4. Interval-scheduled or manually triggered Indian Currency Denomination Recognition
5. Optional Ultrasonic Distance Sensor (HC-SR04 hardware or clearly marked simulation)
6. Prioritized Offline Audio Feedback:
   Priority 1: Immediate obstacle distance warning (<= 2.0m)
   Priority 2: Hazard / obstacle object detection
   Priority 3: Indian currency recognition
   Priority 4: OCR environmental text announcement
"""

import time
from typing import Any, Dict
import numpy as np

from modules.audio_feedback import AudioFeedbackEngine, get_audio_engine
from modules.camera import CameraManager
from modules.currency_detection import CurrencyDetectionModule
from modules.distance_sensor import UltrasonicDistanceSensor
from modules.object_detection import ObjectDetectionModule
from modules.ocr_module import OCRModule
from utils.logger import get_logger, load_config
from utils.performance import PerformanceMonitor
from utils.visualization import draw_currency_detections, draw_detections, draw_ocr_results

logger = get_logger("SafeStep.NavigationEngine")


class SafeStepNavigationEngine:
    """Central coordinator integrating all AI modules, distance sensing, and audio feedback."""

    def __init__(self, config_path: str | None = None, lazy_ocr: bool = True) -> None:
        self.config = load_config(config_path)

        cam_cfg = self.config.get("camera", {})
        obj_cfg = self.config.get("models", {}).get("object_detection", {})
        curr_cfg = self.config.get("models", {}).get("currency_detection", {})
        ocr_cfg = self.config.get("ocr", {})
        dist_cfg = self.config.get("distance_sensor", {})
        aud_cfg = self.config.get("audio", {})
        nav_cfg = self.config.get("navigation", {})

        self.camera = CameraManager(
            source=cam_cfg.get("source", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fallback_to_synthetic=cam_cfg.get("fallback_to_synthetic_stream", True),
            backend=cam_cfg.get("backend", "auto"),
            fps_target=cam_cfg.get("fps_target", 20),
        )
        self.object_detector = ObjectDetectionModule(
            confidence_threshold=obj_cfg.get("confidence_threshold", 0.40),
            iou_threshold=obj_cfg.get("iou_threshold", 0.45),
            img_size=obj_cfg.get("img_size", 416),
        )
        self.currency_detector = CurrencyDetectionModule(
            confidence_threshold=curr_cfg.get("confidence_threshold", 0.45),
            iou_threshold=curr_cfg.get("iou_threshold", 0.45),
            img_size=curr_cfg.get("img_size", 416),
        )
        self.ocr_engine = OCRModule(
            languages=ocr_cfg.get("languages", ["en"]),
            gpu=ocr_cfg.get("gpu", False),
            confidence_threshold=ocr_cfg.get("confidence_threshold", 0.45),
            min_text_length=ocr_cfg.get("min_text_length", 2),
            deduplication_window_seconds=ocr_cfg.get("deduplication_window_seconds", 12.0),
        )
        if not lazy_ocr:
            self.ocr_engine.initialize_reader()

        self.distance_sensor = UltrasonicDistanceSensor(
            mode=dist_cfg.get("mode", "auto"),
            poll_rate_hz=dist_cfg.get("poll_rate_hz", 10.0),
            moving_average_window=dist_cfg.get("moving_average_window", 5),
            safety_threshold_meters=dist_cfg.get("safety_threshold_meters", 2.0),
            critical_threshold_meters=dist_cfg.get("critical_threshold_meters", 1.0),
        )
        self.audio: AudioFeedbackEngine = get_audio_engine()
        self.audio.configure(
            enabled=aud_cfg.get("enabled", True),
            rate=aud_cfg.get("rate", 175),
            volume=aud_cfg.get("volume", 0.95),
            cooldown_seconds=aud_cfg.get("cooldown_seconds", 4.0),
        )
        self.perf = PerformanceMonitor()

        self.ocr_interval_sec = float(nav_cfg.get("ocr_interval_seconds", 5.0))
        self.currency_interval_sec = float(nav_cfg.get("currency_detection_interval_seconds", 3.0))
        self.enable_auto_ocr = True
        self.enable_auto_currency = True
        self.enable_distance_alerts = bool(dist_cfg.get("enabled", True))

        self._last_ocr_ts = 0.0
        self._last_currency_ts = 0.0
        self._latest_ocr_result: Dict[str, Any] = {"detected": False, "combined_text": "", "items": []}
        self._latest_currency_result: Dict[str, Any] = {"detected": False, "notes": []}
        self._running = False

    def start(self) -> bool:
        """Start camera capture and ultrasonic sensor polling."""
        cam_ok = self.camera.start()
        if self.enable_distance_alerts:
            self.distance_sensor.start()
        self._running = cam_ok
        if cam_ok:
            self.audio.speak("SafeStep navigation started.", bypass_cooldown=True)
        return cam_ok

    def stop(self) -> None:
        """Stop navigation loop, camera, and sensor polling."""
        self._running = False
        self.camera.stop()
        self.distance_sensor.stop()
        self.audio.speak("SafeStep navigation stopped.", bypass_cooldown=True)

    def process_frame(
        self,
        frame: np.ndarray | None = None,
        run_ocr_now: bool = False,
        run_currency_now: bool = False,
    ) -> Dict[str, Any]:
        """Execute a complete multi-modal perception and prioritized audio step on one frame."""
        if frame is None:
            ok, frame = self.camera.read_frame()
            if not ok or frame is None:
                return {"error": "No camera frame available"}

        fps = self.perf.record_frame()
        now = time.time()

        # 1. Distance Measurement (Priority 1 check)
        dist_info = self.distance_sensor.get_reading()
        distance_m = dist_info["distance_m"]

        # 2. Real-Time Object & Obstacle Detection (Priority 2 check)
        obj_res = self.object_detector.detect(frame)
        self.perf.record_inference(obj_res.get("inference_ms", 0.0), module="object")

        # 3. Scheduled or Manual Currency Recognition (Priority 3 check)
        should_run_currency = run_currency_now or (
            self.enable_auto_currency and (now - self._last_currency_ts) >= self.currency_interval_sec
        )
        if should_run_currency:
            self._last_currency_ts = now
            self._latest_currency_result = self.currency_detector.detect(frame)
            self.perf.record_inference(self._latest_currency_result.get("inference_ms", 0.0), module="currency")

        # 4. Scheduled or Manual OCR Reading (Priority 4 check)
        should_run_ocr = run_ocr_now or (
            self.enable_auto_ocr and (now - self._last_ocr_ts) >= self.ocr_interval_sec
        )
        if should_run_ocr:
            self._last_ocr_ts = now
            self._latest_ocr_result = self.ocr_engine.extract_text(frame, deduplicate=not run_ocr_now)
            self.perf.record_inference(self._latest_ocr_result.get("inference_ms", 0.0), module="ocr")

        # 5. Alert Priority Determination & Voice Feedback Dispatch
        triggered_alert = ""
        if self.enable_distance_alerts and dist_info["obstacle_within_threshold"]:
            triggered_alert = self.audio.announce_distance(
                distance_m, threshold=dist_info["safety_threshold_m"]
            )
        elif obj_res["hazards"]:
            # Pick highest-confidence hazard (preferring newly tracked hazards)
            hazards_sorted = sorted(
                obj_res["hazards"],
                key=lambda d: (1 if d.get("is_new_track") else 0, d.get("confidence", 0.0)),
                reverse=True,
            )
            top_hazard = hazards_sorted[0]
            triggered_alert = self.audio.announce_obstacle(
                top_hazard["class_name"], position=top_hazard.get("position")
            )
        elif should_run_currency and self._latest_currency_result.get("detected"):
            triggered_alert = self.audio.announce_currency(
                self._latest_currency_result["denomination"]
            )
        elif should_run_ocr and self._latest_ocr_result.get("detected"):
            triggered_alert = self.audio.announce_text(
                self._latest_ocr_result["combined_text"]
            )
        elif obj_res["detections"]:
            top_obj = max(obj_res["detections"], key=lambda d: d.get("confidence", 0.0))
            triggered_alert = self.audio.announce_obstacle(
                top_obj["class_name"], position=top_obj.get("position")
            )

        audio_status = self.audio.get_status()
        self.perf.record_audio_latency(audio_status.get("last_dispatch_latency_ms", 0.0))
        latest_voice = triggered_alert or audio_status.get("latest_announcement", "")

        # 6. Annotate visual frame for dashboard display
        annotated = draw_detections(
            frame,
            obj_res["detections"],
            fps=fps,
            distance_m=distance_m,
            sensor_simulated=dist_info["is_simulated"],
            latest_alert=latest_voice,
        )
        if self._latest_currency_result.get("detected"):
            annotated = draw_currency_detections(annotated, self._latest_currency_result.get("notes", []))
        if self._latest_ocr_result.get("detected"):
            annotated = draw_ocr_results(annotated, self._latest_ocr_result.get("items", []))

        return {
            "annotated_frame": annotated,
            "raw_frame": frame,
            "object_results": obj_res,
            "currency_results": self._latest_currency_result,
            "ocr_results": self._latest_ocr_result,
            "distance_results": dist_info,
            "latest_voice_alert": latest_voice,
            "system_metrics": self.perf.get_system_metrics(),
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """Return comprehensive health status of all SafeStep subsystems."""
        return {
            "camera": self.camera.get_status(),
            "object_model": {
                "loaded": self.object_detector.model_loaded,
                "custom_weights": self.object_detector.using_custom_weights,
                "weights_path": str(self.object_detector.weights_path),
                "backend": self.object_detector.backend,
            },
            "currency_model": {
                "loaded": self.currency_detector.model_loaded,
                "weights_path": str(self.currency_detector.weights_path),
            },
            "ocr_engine": {
                "ready": self.ocr_engine.engine_ready,
                "languages": self.ocr_engine.languages,
            },
            "audio": self.audio.get_status(),
            "distance_sensor": self.distance_sensor.get_reading(),
            "performance": self.perf.get_system_metrics(),
        }
