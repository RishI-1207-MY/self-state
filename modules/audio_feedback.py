"""Offline prioritized text-to-speech audio feedback module using pyttsx3.

Features:
- Priority queue where urgent obstacle/distance alerts preempt lower-priority announcements:
  Priority 1: Immediate obstacle warning (< 2m distance)
  Priority 2: Object and hazard detection
  Priority 3: Indian currency recognition
  Priority 4: OCR text reading
- Cooldown deduplication to prevent repeating the same announcement across consecutive frames
- Non-blocking background daemon worker thread so speech never blocks video processing
- Graceful fallback if audio hardware or pyttsx3 driver is unavailable
- Reusable top-level functions:
  speak(text), announce_obstacle(object_name), announce_distance(distance),
  announce_currency(denomination), announce_text(recognized_text)
"""

from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Deque, Dict, List

from utils.logger import get_logger

logger = get_logger("SafeStep.Audio")

PRIORITY_IMMEDIATE_WARNING = 1
PRIORITY_OBJECT_HAZARD = 2
PRIORITY_CURRENCY = 3
PRIORITY_OCR_TEXT = 4

SPOKEN_DENOMINATIONS: Dict[int | str, str] = {
    10: "Ten rupees",
    20: "Twenty rupees",
    50: "Fifty rupees",
    100: "One hundred rupees",
    200: "Two hundred rupees",
    500: "Five hundred rupees",
    "10": "Ten rupees",
    "20": "Twenty rupees",
    "50": "Fifty rupees",
    "100": "One hundred rupees",
    "200": "Two hundred rupees",
    "500": "Five hundred rupees",
    "10_rupees": "Ten rupees",
    "20_rupees": "Twenty rupees",
    "50_rupees": "Fifty rupees",
    "100_rupees": "One hundred rupees",
    "200_rupees": "Two hundred rupees",
    "500_rupees": "Five hundred rupees",
}


@dataclass(order=True)
class SpeechTask:
    priority: int
    timestamp: float
    text: str = field(compare=False)
    category: str = field(default="general", compare=False)


class AudioFeedbackEngine:
    """Thread-safe prioritized offline TTS manager using pyttsx3."""

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 175,
        volume: float = 0.95,
        cooldown_seconds: float = 4.0,
        urgent_cooldown_seconds: float = 2.0,
        max_queue_size: int = 10,
    ) -> None:
        self.enabled = enabled
        self.rate = int(rate)
        self.volume = float(max(0.0, min(1.0, volume)))
        self.cooldown_seconds = float(cooldown_seconds)
        self.urgent_cooldown_seconds = float(urgent_cooldown_seconds)

        self._queue: queue.PriorityQueue[SpeechTask] = queue.PriorityQueue(maxsize=max_queue_size)
        self._last_spoken_map: Dict[str, float] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=25)
        self._latest_announcement: str = "System ready."
        self._last_dispatch_latency_ms: float = 0.0
        self._driver_available: bool = False
        self._lock = threading.Lock()
        self._running = True

        self._probe_driver()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _probe_driver(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._driver_available = True
            del engine
        except Exception as exc:
            self._driver_available = False
            logger.warning("pyttsx3 audio driver unavailable or headless (%s). Using logged silent TTS fallback.", exc)

    def configure(
        self,
        enabled: bool | None = None,
        rate: int | None = None,
        volume: float | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Update voice settings dynamically."""
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if rate is not None:
                self.rate = int(rate)
            if volume is not None:
                self.volume = float(max(0.0, min(1.0, volume)))
            if cooldown_seconds is not None:
                self.cooldown_seconds = float(cooldown_seconds)

    def enqueue(
        self,
        text: str,
        priority: int = PRIORITY_OBJECT_HAZARD,
        category: str = "general",
        dedup_key: str | None = None,
        bypass_cooldown: bool = False,
    ) -> bool:
        """Enqueue a spoken alert if it passes cooldown deduplication."""
        clean_text = " ".join(str(text).strip().split())
        if not clean_text:
            return False

        now = time.time()
        key = dedup_key or f"{category}:{clean_text.lower()}"
        cooldown = (
            self.urgent_cooldown_seconds
            if priority == PRIORITY_IMMEDIATE_WARNING
            else self.cooldown_seconds
        )

        with self._lock:
            last_time = self._last_spoken_map.get(key, 0.0)
            if not bypass_cooldown and (now - last_time) < cooldown:
                return False
            self._last_spoken_map[key] = now
            self._latest_announcement = clean_text
            self._history.appendleft(
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "text": clean_text,
                    "priority": priority,
                    "category": category,
                }
            )

        task = SpeechTask(priority=priority, timestamp=time.perf_counter(), text=clean_text, category=category)
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(task)
            except Exception:
                return False
        return True

    def _worker_loop(self) -> None:
        while self._running:
            try:
                task = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            dispatch_latency = (time.perf_counter() - task.timestamp) * 1000.0
            with self._lock:
                self._last_dispatch_latency_ms = round(dispatch_latency, 2)

            if self.enabled and self._driver_available:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", self.rate)
                    engine.setProperty("volume", self.volume)
                    engine.say(task.text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as exc:
                    logger.debug("TTS playback warning: %s", exc)
            self._queue.task_done()

    def speak(self, text: str, priority: int = PRIORITY_OBJECT_HAZARD, bypass_cooldown: bool = False) -> bool:
        """Speak arbitrary text with priority queueing."""
        return self.enqueue(text, priority=priority, category="speech", bypass_cooldown=bypass_cooldown)

    def announce_obstacle(self, object_name: str, position: str | None = None) -> str:
        """Format and announce a detected object or obstacle."""
        obj_clean = str(object_name).strip()
        lower = obj_clean.lower()
        pos_phrase = f" on your {position}" if position in ("left", "right") else " ahead"

        if lower == "person":
            msg = f"Person detected{pos_phrase}."
        elif lower in ("car", "bus", "truck", "motorcycle", "bicycle"):
            msg = f"Vehicle ({obj_clean.lower()}) detected{pos_phrase}."
        elif lower == "pole":
            msg = "Pole detected on your path."
        elif lower == "other obstacles":
            msg = "Obstacle ahead, move carefully."
        else:
            msg = f"{obj_clean.capitalize()} detected{pos_phrase}."

        self.enqueue(
            msg,
            priority=PRIORITY_OBJECT_HAZARD,
            category="obstacle",
            dedup_key=f"obstacle:{lower}:{position or 'center'}",
        )
        return msg

    def announce_distance(self, distance: float, threshold: float = 2.0) -> str:
        """Generate an immediate priority-1 warning if obstacle distance is within threshold."""
        dist_val = round(float(distance), 2)
        if dist_val <= 1.0:
            msg = f"Obstacle detected within {dist_val:.1f} meters. Please stop."
        elif dist_val <= threshold:
            msg = "Warning! Obstacle within two meters."
        else:
            msg = f"Path clear at {dist_val:.1f} meters."

        if dist_val <= threshold:
            self.enqueue(
                msg,
                priority=PRIORITY_IMMEDIATE_WARNING,
                category="distance",
                dedup_key="distance:urgent_warning",
            )
        return msg

    def announce_currency(self, denomination: int | str) -> str:
        """Format and announce an Indian currency denomination."""
        spoken = SPOKEN_DENOMINATIONS.get(denomination, f"{denomination} rupees")
        msg = f"Indian currency detected. {spoken}."
        self.enqueue(
            msg,
            priority=PRIORITY_CURRENCY,
            category="currency",
            dedup_key=f"currency:{denomination}",
        )
        return msg

    def announce_text(self, recognized_text: str) -> str:
        """Format and announce OCR-extracted environmental text."""
        cleaned = " ".join(str(recognized_text).strip().split())
        if not cleaned:
            return ""
        msg = f"Text detected. Reading: {cleaned}."
        self.enqueue(
            msg,
            priority=PRIORITY_OCR_TEXT,
            category="ocr",
            dedup_key=f"ocr:{cleaned.lower()}",
        )
        return msg

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "driver_available": self._driver_available,
                "rate": self.rate,
                "volume": self.volume,
                "latest_announcement": self._latest_announcement,
                "last_dispatch_latency_ms": self._last_dispatch_latency_ms,
                "history": list(self._history),
            }

    def shutdown(self) -> None:
        self._running = False


# Singleton instance & required reusable module-level functions
_default_engine: AudioFeedbackEngine | None = None


def get_audio_engine() -> AudioFeedbackEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = AudioFeedbackEngine()
    return _default_engine


def speak(text: str, bypass_cooldown: bool = False) -> bool:
    """Speak arbitrary text through the offline TTS engine."""
    return get_audio_engine().speak(text, bypass_cooldown=bypass_cooldown)


def announce_obstacle(object_name: str, position: str | None = None) -> str:
    """Announce a detected obstacle or object."""
    return get_audio_engine().announce_obstacle(object_name, position=position)


def announce_distance(distance: float, threshold: float = 2.0) -> str:
    """Announce an ultrasonic distance warning when within safety threshold."""
    return get_audio_engine().announce_distance(distance, threshold=threshold)


def announce_currency(denomination: int | str) -> str:
    """Announce a detected Indian currency note denomination."""
    return get_audio_engine().announce_currency(denomination)


def announce_text(recognized_text: str) -> str:
    """Announce environmental text recognized via OCR."""
    return get_audio_engine().announce_text(recognized_text)
