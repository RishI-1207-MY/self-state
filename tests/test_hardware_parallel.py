"""Unit and integration tests for SafeStep Raspberry Pi 4 parallel hardware modules and configs."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.picamera2_interface import check_picamera2_availability
from hardware.test_audio_hw import run_audio_hardware_test
from hardware.test_hcsr04_hw import run_hcsr04_hardware_test
from hardware.test_picamera2_hw import run_picamera2_hardware_test
from utils.logger import load_config


def test_windows_and_rpi_configurations_load_cleanly() -> None:
    win_cfg = load_config(profile="windows")
    rpi_cfg = load_config(profile="rpi")

    assert win_cfg["profile"] == "windows_development"
    assert win_cfg["distance_sensor"]["mode"] == "simulation"
    assert win_cfg["camera"]["backend"] == "opencv"

    assert rpi_cfg["profile"] == "raspberry_pi_4_deployment"
    assert rpi_cfg["distance_sensor"]["mode"] == "hardware"
    assert rpi_cfg["distance_sensor"]["trigger_pin"] == 23
    assert rpi_cfg["distance_sensor"]["echo_pin"] == 24
    assert rpi_cfg["camera"]["backend"] == "picamera2"


def test_hardware_scripts_mark_pending_on_laptop() -> None:
    picam_res = run_picamera2_hardware_test(num_frames=2)
    hcsr_res = run_hcsr04_hardware_test(samples=2, allow_simulation=False)
    aud_res = run_audio_hardware_test(speak_out_loud=False)

    diag = check_picamera2_availability()
    if not diag["is_raspberry_pi"]:
        assert picam_res["status"] == "PENDING_PHYSICAL_PI_EXECUTION"
        assert picam_res["is_simulated"] is False
        assert hcsr_res["status"] == "PENDING_PHYSICAL_PI_EXECUTION"
        assert hcsr_res["is_simulated"] is False
        assert "PENDING" in aud_res["status"]

    # Also verify explicit simulation flag for HC-SR04 marks is_simulated=True
    hcsr_sim = run_hcsr04_hardware_test(samples=3, allow_simulation=True)
    if not diag["is_raspberry_pi"]:
        assert hcsr_sim["is_simulated"] is True
        assert "SIMULATION" in hcsr_sim["status"]
