"""Raspberry Pi hardware detection, GPIO diagnostics, and setup helper for SafeStep."""

import platform
from typing import Any, Dict
from hardware.gpio_config import DEFAULT_GPIO_CONFIG, UltrasonicGPIOConfig
from utils.logger import get_logger

logger = get_logger("SafeStep.Hardware")


def is_raspberry_pi() -> bool:
    """Detect if the current system is running on Raspberry Pi hardware."""
    try:
        machine = platform.machine().lower()
        if not ("arm" in machine or "aarch64" in machine):
            return False
        with open("/proc/device-tree/model", "r", encoding="utf-8", errors="ignore") as f:
            model_str = f.read().lower()
            return "raspberry pi" in model_str
    except Exception:
        return False


def check_gpio_availability() -> Dict[str, Any]:
    """Check whether RPi.GPIO or gpiozero is available and return hardware diagnostics."""
    rpi_detected = is_raspberry_pi()
    rpi_gpio_installed = False
    gpiozero_installed = False

    try:
        import RPi.GPIO  # type: ignore # noqa: F401
        rpi_gpio_installed = True
    except ImportError:
        pass

    try:
        import gpiozero  # type: ignore # noqa: F401
        gpiozero_installed = True
    except ImportError:
        pass

    hardware_ready = rpi_detected and (rpi_gpio_installed or gpiozero_installed)
    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "is_raspberry_pi": rpi_detected,
        "rpi_gpio_installed": rpi_gpio_installed,
        "gpiozero_installed": gpiozero_installed,
        "hardware_ready": hardware_ready,
        "trigger_pin_bcm": DEFAULT_GPIO_CONFIG.trigger_pin,
        "echo_pin_bcm": DEFAULT_GPIO_CONFIG.echo_pin,
        "fallback_mode": "simulation" if not hardware_ready else "hardware",
    }


def initialize_hc_sr04_pins(config: UltrasonicGPIOConfig = DEFAULT_GPIO_CONFIG) -> bool:
    """Initialize Raspberry Pi GPIO pins for the HC-SR04 sensor if hardware is available."""
    status = check_gpio_availability()
    if not status["hardware_ready"]:
        logger.info(
            "Raspberry Pi GPIO hardware not detected (OS=%s, Arch=%s). "
            "SafeStep will run distance sensor in clearly marked SIMULATION mode.",
            status["os"],
            status["architecture"],
        )
        return False

    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM if config.bcm_mode else GPIO.BOARD)
        GPIO.setup(config.trigger_pin, GPIO.OUT)
        GPIO.setup(config.echo_pin, GPIO.IN)
        GPIO.output(config.trigger_pin, False)
        logger.info(
            "Initialized HC-SR04 GPIO pins (TRIG=BCM %d, ECHO=BCM %d).",
            config.trigger_pin,
            config.echo_pin,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to initialize RPi.GPIO pins: %s", exc)
        return False


def get_wiring_guide() -> str:
    """Return a formatted wiring guide for connecting the HC-SR04 sensor to a Raspberry Pi."""
    return (
        "HC-SR04 to Raspberry Pi 4B / 5 Wiring Guide (BCM Numbering):\n"
        "  1. VCC  -> Physical Pin 2  (5V Power)\n"
        "  2. TRIG -> Physical Pin 16 (GPIO 23)\n"
        "  3. ECHO -> Physical Pin 18 (GPIO 24) MUST use voltage divider (1kΩ + 2kΩ) to step 5V down to 3.3V\n"
        "  4. GND  -> Physical Pin 6  (Ground)\n"
    )


if __name__ == "__main__":
    diag = check_gpio_availability()
    print("=== SafeStep Hardware Diagnostics ===")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print("\n" + get_wiring_guide())
