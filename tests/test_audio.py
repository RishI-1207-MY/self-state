"""Unit and integration tests for SafeStep Audio Feedback and Ultrasonic Distance Sensor."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.audio_feedback import (
    AudioFeedbackEngine,
    announce_currency,
    announce_distance,
    announce_obstacle,
    announce_text,
    speak,
)
from modules.distance_sensor import UltrasonicDistanceSensor


def test_audio_priority_and_cooldown() -> None:
    engine = AudioFeedbackEngine(enabled=False, cooldown_seconds=5.0, urgent_cooldown_seconds=2.0)

    ok1 = engine.enqueue("Person detected ahead.", priority=2, category="obstacle", dedup_key="test_person")
    ok2 = engine.enqueue("Person detected ahead.", priority=2, category="obstacle", dedup_key="test_person")
    assert ok1 is True
    assert ok2 is False  # Blocked by cooldown

    msg_obs = announce_obstacle("Pole")
    assert "Pole detected on your path." == msg_obs

    msg_curr = announce_currency(500)
    assert "Five hundred rupees" in msg_curr

    msg_txt = announce_text("Room Number 204")
    assert msg_txt == "Text detected. Reading: Room Number 204."

    msg_dist = announce_distance(1.5, threshold=2.0)
    assert "Warning! Obstacle within two meters." == msg_dist
    assert speak("Test speech", bypass_cooldown=True) is True
    engine.shutdown()


def test_ultrasonic_sensor_simulation_mode() -> None:
    sensor = UltrasonicDistanceSensor(mode="simulation", safety_threshold_meters=2.0)
    assert sensor.is_simulated is True

    sensor.set_simulated_distance(1.4)
    reading = sensor.poll_once()
    assert reading["is_simulated"] is True
    assert "SIMULATION" in reading["mode_label"]
    assert reading["obstacle_within_threshold"] is True
    assert abs(reading["distance_m"] - 1.4) < 0.05
