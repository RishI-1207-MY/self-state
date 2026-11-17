"""Independent Hardware Test Script 1: Raspberry Pi Camera Module (CSI / Picamera2).

Verifies:
1. Raspberry Pi hardware detection & `libcamera` / `rpicam-hello` availability
2. `Picamera2` initialization and CSI ribbon sensor detection
3. Capturing 10 live frames at 640x480 and measuring CSI capture FPS
4. Saving a sample verification snapshot to `logs/picamera2_hardware_test.jpg`

If executed on the Windows development laptop (before transferring to the Pi),
this script exits cleanly and marks the hardware test status as `PENDING_PHYSICAL_PI_EXECUTION`
(never fabricating physical CSI camera measurements).
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.picamera2_interface import PiCamera2HardwareStream, check_picamera2_availability
from utils.logger import get_logger

logger = get_logger("SafeStep.HWTest.Picamera2")


def run_picamera2_hardware_test(num_frames: int = 10, width: int = 640, height: int = 480) -> Dict[str, Any]:
    """Run physical CSI camera verification or record PENDING status if not on Raspberry Pi."""
    diag = check_picamera2_availability()
    report_path = PROJECT_ROOT / "evaluation" / "results" / "hw_test_picamera2.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not diag["is_raspberry_pi"]:
        result = {
            "component": "Raspberry Pi Camera Module (CSI / Picamera2)",
            "status": "PENDING_PHYSICAL_PI_EXECUTION",
            "is_simulated": False,
            "reason": "Executed on development laptop (non-Raspberry Pi host). Run this script on Raspberry Pi 4 after connecting the CSI ribbon cable.",
            "diagnostics": diag,
        }
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Picamera2 Hardware Test Status: PENDING_PHYSICAL_PI_EXECUTION (Currently on Windows laptop).")
        return result

    if not diag["picamera2_installed"]:
        result = {
            "component": "Raspberry Pi Camera Module (CSI / Picamera2)",
            "status": "FAILED_MISSING_PICAMERA2",
            "is_simulated": False,
            "reason": "python3-picamera2 is not installed. Run `sudo apt install -y python3-picamera2 rpicam-apps`.",
            "diagnostics": diag,
        }
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.error("Picamera2 package missing on Raspberry Pi.")
        return result

    stream = PiCamera2HardwareStream(width=width, height=height, fps=15)
    if not stream.start():
        result = {
            "component": "Raspberry Pi Camera Module (CSI / Picamera2)",
            "status": "FAILED_CSI_INIT",
            "is_simulated": False,
            "reason": "Picamera2 failed to open CSI camera. Check ribbon cable orientation on the CSI port and run `rpicam-hello --list-cameras`.",
            "diagnostics": diag,
        }
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    captured = 0
    t0 = time.perf_counter()
    last_frame = None
    for _ in range(num_frames):
        ok, frame = stream.read_bgr_frame()
        if ok and frame is not None:
            captured += 1
            last_frame = frame
        time.sleep(0.02)
    elapsed = max(0.001, time.perf_counter() - t0)
    stream.stop()

    snapshot_path = PROJECT_ROOT / "logs" / "picamera2_hardware_test.jpg"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if last_frame is not None:
        cv2.imwrite(str(snapshot_path), last_frame)

    passed = captured > 0
    result = {
        "component": "Raspberry Pi Camera Module (CSI / Picamera2)",
        "status": "PASSED_ON_HARDWARE" if passed else "FAILED_ZERO_FRAMES",
        "is_simulated": False,
        "frames_requested": num_frames,
        "frames_captured": captured,
        "resolution": f"{width}x{height}",
        "measured_fps": round(captured / elapsed, 2),
        "sample_snapshot": str(snapshot_path) if last_frame is not None else None,
        "diagnostics": diag,
    }
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Picamera2 CSI Hardware Test Result: %s (%d/%d frames)", result["status"], captured, num_frames)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Raspberry Pi CSI Camera via Picamera2")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    res = run_picamera2_hardware_test(num_frames=args.frames)
    print(json.dumps(res, indent=2))
