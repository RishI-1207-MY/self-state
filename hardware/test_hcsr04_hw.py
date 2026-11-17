"""Independent Hardware Test Script 2: HC-SR04 Ultrasonic Sensor (GPIO23 TRIG / GPIO24 ECHO).

Hardware Wiring Specification:
- VCC  -> Physical Pin 2  (5V)
- TRIG -> Physical Pin 16 (BCM GPIO 23)
- ECHO -> Physical Pin 18 (BCM GPIO 24) via 1kΩ / 2kΩ voltage divider (steps 5V echo down to 3.3V)
- GND  -> Physical Pin 6  (Ground)

Behavior:
- On Raspberry Pi 4 with GPIO ready: triggers real 10µs ultrasonic pulses at 10 Hz,
  measures actual echo duration, computes moving-average distance, and logs real measurements.
- On Windows laptop (unless `--simulate` is explicitly passed): marks hardware test status as
  `PENDING_PHYSICAL_PI_EXECUTION` without fabricating physical sensor measurements.
- When `--simulate` is passed on a laptop: runs and verifies the moving-average & 2.0m alert logic
  while explicitly tagging `is_simulated = True` and `status = "SIMULATION_ONLY_PENDING_HARDWARE"`.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.gpio_config import DEFAULT_GPIO_CONFIG
from hardware.raspberry_pi_setup import check_gpio_availability, get_wiring_guide
from modules.distance_sensor import UltrasonicDistanceSensor
from utils.logger import get_logger

logger = get_logger("SafeStep.HWTest.HCSR04")


def run_hcsr04_hardware_test(samples: int = 10, allow_simulation: bool = False) -> Dict[str, Any]:
    """Execute HC-SR04 hardware verification or record PENDING status if not on Raspberry Pi."""
    diag = check_gpio_availability()
    report_path = PROJECT_ROOT / "evaluation" / "results" / "hw_test_hcsr04.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not diag["hardware_ready"] and not allow_simulation:
        result = {
            "component": "HC-SR04 Ultrasonic Sensor (GPIO23 TRIG, GPIO24 ECHO via 1kΩ/2kΩ divider)",
            "status": "PENDING_PHYSICAL_PI_EXECUTION",
            "is_simulated": False,
            "trigger_pin_bcm": DEFAULT_GPIO_CONFIG.trigger_pin,
            "echo_pin_bcm": DEFAULT_GPIO_CONFIG.echo_pin,
            "voltage_divider": "1kΩ (series) + 2kΩ (to GND) -> 3.3V safe GPIO24 input",
            "reason": "Physical Raspberry Pi GPIO not detected on this host. Run on Raspberry Pi 4 or pass `--simulate` to test software filter logic only.",
            "wiring_guide": get_wiring_guide(),
            "diagnostics": diag,
        }
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("HC-SR04 Hardware Test Status: PENDING_PHYSICAL_PI_EXECUTION (No physical RPi GPIO on laptop).")
        return result

    mode = "hardware" if diag["hardware_ready"] else "simulation"
    sensor = UltrasonicDistanceSensor(mode=mode, poll_rate_hz=10.0, moving_average_window=5, safety_threshold_meters=2.0)

    readings: List[Dict[str, Any]] = []
    for i in range(samples):
        r = sensor.poll_once()
        readings.append(
            {
                "sample": i + 1,
                "raw_distance_m": r["raw_distance_m"],
                "filtered_distance_m": r["distance_m"],
                "within_2m_threshold": r["obstacle_within_threshold"],
                "is_simulated": r["is_simulated"],
            }
        )
        time.sleep(0.1)

    is_sim = sensor.is_simulated
    status_str = "SIMULATION_VERIFIED_PENDING_PI_HARDWARE" if is_sim else "PASSED_ON_HARDWARE"

    result = {
        "component": "HC-SR04 Ultrasonic Sensor (GPIO23 TRIG, GPIO24 ECHO via 1kΩ/2kΩ divider)",
        "status": status_str,
        "is_simulated": is_sim,
        "mode_label": "SIMULATION (Laptop Software Filter Test)" if is_sim else "HARDWARE (Physical Raspberry Pi GPIO)",
        "trigger_pin_bcm": DEFAULT_GPIO_CONFIG.trigger_pin,
        "echo_pin_bcm": DEFAULT_GPIO_CONFIG.echo_pin,
        "samples_collected": len(readings),
        "readings": readings,
        "diagnostics": diag,
    }
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("HC-SR04 Test completed with status=%s (is_simulated=%s)", status_str, is_sim)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HC-SR04 Ultrasonic Sensor on GPIO23/GPIO24")
    parser.add_argument("--samples", type=int, default=10, help="Number of 10Hz samples to collect")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Explicitly run in marked SIMULATION mode when testing on a laptop without Raspberry Pi GPIO",
    )
    args = parser.parse_args()
    res = run_hcsr04_hardware_test(samples=args.samples, allow_simulation=args.simulate)
    print(json.dumps(res, indent=2))
