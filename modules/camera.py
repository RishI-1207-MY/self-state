"""Camera capture manager for SafeStep supporting Picamera2 (CSI), OpenCV webcam, and demo fallback.

Supports:
1. Raspberry Pi Camera Module via CSI (`hardware/picamera2_interface.py`) when `backend="picamera2"` or `"auto"` on RPi
2. Standard USB / laptop webcam via OpenCV `VideoCapture`
3. Explicitly marked DEMO/SYNTHETIC fallback stream when physical camera hardware is not attached
"""

import threading
import time
from typing import Any, Dict, Tuple
import cv2
import numpy as np

from hardware.picamera2_interface import PiCamera2HardwareStream, check_picamera2_availability
from utils.logger import get_logger

logger = get_logger("SafeStep.Camera")


class CameraManager:
    """Thread-safe camera capture manager with Picamera2 CSI, OpenCV, and fallback support."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fallback_to_synthetic: bool = True,
        backend: str = "auto",
        fps_target: int = 20,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fallback_to_synthetic = fallback_to_synthetic
        self.backend = backend.lower()
        self.fps_target = max(5, int(fps_target))

        self._picam_stream: PiCamera2HardwareStream | None = None
        self._cap: cv2.VideoCapture | None = None
        self._active_driver: str = "uninitialized"
        self._running = False
        self._using_fallback = False
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_index = 0
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Open camera source (Picamera2 CSI or OpenCV) and start background frame reader thread."""
        if self._running:
            return True

        # 1. Try Picamera2 CSI hardware if requested or auto-detected on Raspberry Pi
        if self.backend in ("picamera2", "auto"):
            csi_status = check_picamera2_availability()
            if self.backend == "picamera2" or csi_status["csi_hardware_ready"]:
                picam = PiCamera2HardwareStream(width=self.width, height=self.height, fps=self.fps_target)
                if picam.start():
                    ok, test_frame = picam.read_bgr_frame()
                    if ok and test_frame is not None:
                        self._picam_stream = picam
                        self._active_driver = "picamera2_csi"
                        self._using_fallback = False
                        self._latest_frame = cv2.resize(test_frame, (self.width, self.height))
                        logger.info("CSI Camera active via Picamera2 (%dx%d).", self.width, self.height)

        # 2. Try OpenCV VideoCapture if Picamera2 CSI was not started
        if self._picam_stream is None and self.backend in ("opencv", "auto", "picamera2"):
            try:
                if isinstance(self.source, int):
                    cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(self.source)
                else:
                    cap = cv2.VideoCapture(self.source)

                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        self._cap = cap
                        self._active_driver = "opencv_webcam"
                        self._using_fallback = False
                        self._latest_frame = cv2.resize(test_frame, (self.width, self.height))
                        logger.info("Camera source %s opened via OpenCV (%dx%d).", self.source, self.width, self.height)
                    else:
                        cap.release()
                        self._cap = None
                else:
                    cap.release()
                    self._cap = None
            except Exception as exc:
                logger.warning("Error opening camera %s: %s", self.source, exc)
                self._cap = None

        # 3. Fallback to clearly marked synthetic demo stream if no physical camera is available
        if self._picam_stream is None and self._cap is None:
            if not self.fallback_to_synthetic:
                logger.error("Physical camera unavailable and synthetic fallback is disabled.")
                return False
            self._using_fallback = True
            self._active_driver = "synthetic_demo_fallback"
            self._latest_frame = self._generate_demo_frame(0)
            logger.info("Physical camera unavailable. Activated clearly marked DEMO/SYNTHETIC camera stream.")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self) -> None:
        while self._running:
            if not self._using_fallback and self._picam_stream is not None:
                ok, frame = self._picam_stream.read_bgr_frame()
                if ok and frame is not None and frame.size > 0:
                    resized = cv2.resize(frame, (self.width, self.height))
                    with self._lock:
                        self._latest_frame = resized
                        self._frame_index += 1
                else:
                    time.sleep(0.03)
            elif not self._using_fallback and self._cap is not None:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    resized = cv2.resize(frame, (self.width, self.height))
                    with self._lock:
                        self._latest_frame = resized
                        self._frame_index += 1
                else:
                    if self.fallback_to_synthetic:
                        self._using_fallback = True
                        self._active_driver = "synthetic_demo_fallback"
                    time.sleep(0.05)
            else:
                frame = self._generate_demo_frame(self._frame_index)
                with self._lock:
                    self._latest_frame = frame
                    self._frame_index += 1
                time.sleep(1.0 / 20.0)

    def _generate_demo_frame(self, step: int) -> np.ndarray:
        """Render a dynamic indoor/outdoor navigation frame with text and obstacle for demo/testing."""
        w, h = self.width, self.height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.42)
        canvas[:horizon, :] = (195, 175, 145)
        canvas[horizon:, :] = (90, 95, 100)

        cv2.line(canvas, (w // 2 - 50, horizon), (20, h), (210, 210, 210), 2)
        cv2.line(canvas, (w // 2 + 50, horizon), (w - 20, h), (210, 210, 210), 2)

        cv2.rectangle(canvas, (30, 45), (255, 105), (245, 245, 245), -1)
        cv2.rectangle(canvas, (30, 45), (255, 105), (30, 30, 30), 2)
        cv2.putText(canvas, "ROOM 204", (48, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 15, 15), 3, cv2.LINE_AA)

        offset_x = int(45 * np.sin(step * 0.08))
        px1 = w // 2 - 45 + offset_x
        py1 = horizon - 15
        px2 = px1 + 90
        py2 = int(h * 0.86)
        cv2.circle(canvas, ((px1 + px2) // 2, py1 + 22), 18, (140, 175, 215), -1)
        cv2.rectangle(canvas, (px1 + 15, py1 + 42), (px2 - 15, py1 + 135), (45, 110, 205), -1)
        cv2.rectangle(canvas, (px1 + 18, py1 + 135), ((px1 + px2) // 2 - 3, py2), (40, 50, 75), -1)
        cv2.rectangle(canvas, ((px1 + px2) // 2 + 3, py1 + 135), (px2 - 18, py2), (40, 50, 75), -1)

        cv2.rectangle(canvas, (w - 115, horizon - 40), (w - 90, int(h * 0.88)), (135, 140, 145), -1)
        cv2.rectangle(canvas, (w - 115, int(h * 0.65)), (w - 90, int(h * 0.73)), (0, 215, 255), -1)

        cv2.putText(
            canvas,
            "DEMO STREAM (SIMULATION - NO PHYSICAL CAMERA ATTACHED)",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def read_frame(self) -> Tuple[bool, np.ndarray | None]:
        """Return the latest captured frame."""
        if not self._running:
            if not self.start():
                return False, None
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def stop(self) -> None:
        """Stop capture thread and release hardware camera."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._picam_stream is not None:
            self._picam_stream.stop()
            self._picam_stream = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def get_status(self) -> Dict[str, Any]:
        """Return camera connection status and active driver."""
        return {
            "running": self._running,
            "configured_backend": self.backend,
            "active_driver": self._active_driver,
            "source": self.source,
            "resolution": f"{self.width}x{self.height}",
            "using_demo_fallback": self._using_fallback,
            "frames_captured": self._frame_index,
            "status_label": (
                "Connected (Raspberry Pi CSI Picamera2)"
                if self._active_driver == "picamera2_csi"
                else (
                    "Connected (Hardware Webcam)"
                    if (self._running and not self._using_fallback)
                    else ("Active (SIMULATION Demo Stream)" if self._running else "Stopped")
                )
            ),
        }
