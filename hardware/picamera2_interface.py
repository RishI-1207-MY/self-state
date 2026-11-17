"""Dedicated Raspberry Pi Camera Module (CSI / Picamera2) hardware interface for SafeStep.

Isolated inside `hardware/picamera2_interface.py` so Windows laptop development
never requires `picamera2` or `libcamera` to be installed.

Hardware Target:
- Raspberry Pi 4 Model B (4GB RAM)
- Raspberry Pi Camera Module connected through the CSI ribbon port
- Supports Picamera2 native capture (`Picamera2.capture_array()`) and `rpicam-still` / `libcamera` diagnostics.
"""

from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, Tuple
import cv2
import numpy as np

from hardware.raspberry_pi_setup import is_raspberry_pi
from utils.logger import get_logger

logger = get_logger("SafeStep.Hardware.Picamera2")


def check_picamera2_availability() -> Dict[str, Any]:
    """Inspect whether Picamera2, libcamera, and a CSI camera sensor are available."""
    on_rpi = is_raspberry_pi()
    picamera2_installed = False
    rpicam_cli = shutil.which("rpicam-hello") or shutil.which("libcamera-hello")
    detected_cameras = []

    try:
        import picamera2  # type: ignore # noqa: F401
        from picamera2 import Picamera2  # type: ignore

        picamera2_installed = True
        try:
            cam_info = Picamera2.global_camera_info()
            if cam_info:
                detected_cameras = list(cam_info)
        except Exception:
            pass
    except ImportError:
        pass

    return {
        "is_raspberry_pi": on_rpi,
        "picamera2_installed": picamera2_installed,
        "libcamera_cli_path": rpicam_cli,
        "detected_csi_cameras": detected_cameras,
        "csi_hardware_ready": bool(on_rpi and picamera2_installed and len(detected_cameras) > 0),
    }


class PiCamera2HardwareStream:
    """CSI Camera stream wrapper using Picamera2 on Raspberry Pi 4."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 15) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._picam2: Any = None
        self._running = False

    def start(self) -> bool:
        """Initialize and start Picamera2 CSI video configuration."""
        if self._running and self._picam2 is not None:
            return True

        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam2 = Picamera2()
            config = self._picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameDurationLimits": (int(1_000_000 / self.fps), int(1_000_000 / self.fps))},
            )
            self._picam2.configure(config)
            self._picam2.start()
            time.sleep(0.25)  # Allow CSI sensor auto-exposure / AWB warmup
            self._running = True
            logger.info("Started Picamera2 CSI stream (%dx%d @ %d FPS).", self.width, self.height, self.fps)
            return True
        except Exception as exc:
            logger.info("Picamera2 CSI stream not available on this host (%s).", exc)
            self._picam2 = None
            self._running = False
            return False

    def read_bgr_frame(self) -> Tuple[bool, np.ndarray | None]:
        """Capture a single frame from Picamera2 and return as OpenCV BGR ndarray."""
        if not self._running or self._picam2 is None:
            return False, None
        try:
            arr = self._picam2.capture_array("main")
            if arr is None or arr.size == 0:
                return False, None
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                # Convert RGB888 from Picamera2 to OpenCV BGR
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return True, bgr
            return True, arr
        except Exception as exc:
            logger.warning("Picamera2 frame read error: %s", exc)
            return False, None

    def stop(self) -> None:
        """Stop and close the Picamera2 instance."""
        self._running = False
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None
