"""Raspberry Pi GPIO pin mappings and hardware constants for SafeStep."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UltrasonicGPIOConfig:
    """BCM GPIO pin assignment and timing constants for HC-SR04 Ultrasonic Sensor."""

    bcm_mode: bool = True
    trigger_pin: int = 23       # Physical Pin 16 (GPIO 23)
    echo_pin: int = 24          # Physical Pin 18 (GPIO 24) - via 1k/2k Ohm voltage divider to 3.3V
    pulse_duration_sec: float = 0.00001  # 10 microseconds trigger pulse
    speed_of_sound_m_s: float = 343.0    # Speed of sound at ~20°C in m/s
    timeout_sec: float = 0.04            # 40 ms max wait (~6.8 meters max range)
    min_valid_distance_m: float = 0.02   # 2 cm minimum valid HC-SR04 reading
    max_valid_distance_m: float = 4.50   # 4.5 m maximum reliable HC-SR04 reading


DEFAULT_GPIO_CONFIG = UltrasonicGPIOConfig()
