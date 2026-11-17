"""Ultrasonic distance measurement module (HC-SR04) with Raspberry Pi GPIO and laptop simulation mode.

Features:
- Real HC-SR04 hardware measurement on Raspberry Pi via BCM GPIO trigger/echo pulses
- 10 Hz background polling with moving-average filtering (configurable window)
- Sensor timeout and out-of-range reading rejection
- Configurable safety threshold (default 2.0 meters)
- Clearly marked simulation mode when running on a standard Windows/Linux laptop
  (never misrepresents simulated distance as hardware-measured distance)
"""

from collections import deque
import math
import threading
import time
from typing import Any, Deque, Dict

from hardware.gpio_config import DEFAULT_GPIO_CONFIG, UltrasonicGPIOConfig
from hardware.raspberry_pi_setup import check_gpio_availability, initialize_hc_sr04_pins
from utils.logger import get_logger

logger = get_logger("SafeStep.DistanceSensor")


class UltrasonicDistanceSensor:
    """HC-SR04 Ultrasonic Sensor controller with 10 Hz polling and moving-average filter."""

    def __init__(
        self,
        mode: str = "auto",
        poll_rate_hz: float = 10.0,
        moving_average_window: int = 5,
        safety_threshold_meters: float = 2.0,
        critical_threshold_meters: float = 1.0,
        gpio_config: UltrasonicGPIOConfig = DEFAULT_GPIO_CONFIG,
    ) -> None:
        self.poll_rate_hz = max(1.0, float(poll_rate_hz))
        self.safety_threshold_meters = float(safety_threshold_meters)
        self.critical_threshold_meters = float(critical_threshold_meters)
        self.gpio_config = gpio_config
        self.readings_window: Deque[float] = deque(maxlen=max(1, int(moving_average_window)))

        hw_diag = check_gpio_availability()
        if mode == "hardware" and hw_diag["hardware_ready"]:
            self.is_simulated = not initialize_hc_sr04_pins(gpio_config)
        elif mode == "auto" and hw_diag["hardware_ready"]:
            self.is_simulated = not initialize_hc_sr04_pins(gpio_config)
        else:
            self.is_simulated = True

        self._simulated_Override_distance: float | None = None
        self._latest_filtered_distance: float = 2.65
        self._latest_raw_distance: float = 2.65
        self._invalid_reading_count: int = 0
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_time = time.time()

    def start(self) -> None:
        """Start background 10 Hz sensor polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def set_simulated_distance(self, distance_meters: float | None) -> None:
        """Allow manual override of simulated distance from UI slider or test suite."""
        with self._lock:
            self._simulated_Override_distance = (
                float(distance_meters) if distance_meters is not None else None
            )

    def _measure_hardware_once(self) -> float | None:
        """Trigger a 10us pulse on HC-SR04 and measure echo pulse duration."""
        try:
            import RPi.GPIO as GPIO  # type: ignore

            cfg = self.gpio_config
            GPIO.output(cfg.trigger_pin, True)
            time.sleep(cfg.pulse_duration_sec)
            GPIO.output(cfg.trigger_pin, False)

            t_Init = time.perf_counter()
            pulse_start = t_Init
            while GPIO.input(cfg.echo_pin) == 0:
                pulse_start = time.perf_counter()
                if (pulse_start - t_Init) > cfg.timeout_sec:
                    return None

            pulse_end = pulse_start
            while GPIO.input(cfg.echo_pin) == 1:
                pulse_end = time.perf_counter()
                if (pulse_end - pulse_start) > cfg.timeout_sec:
                    return None

            duration = pulse_end - pulse_start
            distance_m = (duration * cfg.speed_of_sound_m_s) / 2.0
            if cfg.min_valid_distance_m <= distance_m <= cfg.max_valid_distance_m:
                return round(distance_m, 3)
            return None
        except Exception:
            return None

    def _measure_simulated_once(self) -> float:
        """Generate a realistic simulated distance trajectory when running without RPi GPIO."""
        with self._lock:
            if self._simulated_Override_distance is not None:
                return round(max(0.15, min(4.5, self._simulated_Override_distance)), 3)
        elapsed = time.time() - self._start_time
        # Smooth oscillation between 1.35m and 3.40m to demonstrate threshold alerts in simulation
        sim_val = 2.35 + 1.05 * math.sin(elapsed * 0.45)
        return round(sim_val, 3)

    def poll_once(self) -> Dict[str, Any]:
        """Perform a single measurement, update moving average filter, and return status dict."""
        if self.is_simulated:
            raw_dist = self._measure_simulated_once()
        else:
            raw_dist = self._measure_hardware_once()

        with self._lock:
            if raw_dist is None or not (
                self.gpio_config.min_valid_distance_m <= raw_dist <= self.gpio_config.max_valid_distance_m
            ):
                self._invalid_reading_count += 1
            else:
                self._latest_raw_distance = raw_dist
                self.readings_window.append(raw_dist)
                self._latest_filtered_distance = round(
                    sum(self.readings_window) / len(self.readings_window), 3
                )

            dist = self._latest_filtered_distance
            within_safety = dist <= self.safety_threshold_meters
            critical = dist <= self.critical_threshold_meters

            return {
                "distance_m": dist,
                "raw_distance_m": self._latest_raw_distance,
                "is_simulated": self.is_simulated,
                "mode_label": "SIMULATION (No Raspberry Pi GPIO)" if self.is_simulated else "HARDWARE (HC-SR04 GPIO)",
                "obstacle_within_threshold": within_safety,
                "critical_stop_required": critical,
                "safety_threshold_m": self.safety_threshold_meters,
                "invalid_readings": self._invalid_reading_count,
            }

    def _poll_loop(self) -> None:
        interval = 1.0 / self.poll_rate_hz
        while self._running:
            t0 = time.perf_counter()
            self.poll_once()
            elapsed = time.perf_counter() - t0
            sleep_for = max(0.005, interval - elapsed)
            time.sleep(sleep_for)

    def get_reading(self) -> Dict[str, Any]:
        """Return latest filtered reading without blocking."""
        if not self._running:
            return self.poll_once()
        with self._lock:
            dist = self._latest_filtered_distance
            return {
                "distance_m": dist,
                "raw_distance_m": self._latest_raw_distance,
                "is_simulated": self.is_simulated,
                "mode_label": "SIMULATION (No Raspberry Pi GPIO)" if self.is_simulated else "HARDWARE (HC-SR04 GPIO)",
                "obstacle_within_threshold": dist <= self.safety_threshold_meters,
                "critical_stop_required": dist <= self.critical_threshold_meters,
                "safety_threshold_m": self.safety_threshold_meters,
                "invalid_readings": self._invalid_reading_count,
            }
