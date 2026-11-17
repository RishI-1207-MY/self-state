"""System performance, latency, and resource monitoring utilities for SafeStep."""

from collections import deque
import time
from typing import Any, Dict, Deque
import psutil


class PerformanceMonitor:
    """Tracks FPS, inference latency, audio latency, CPU usage, and RAM consumption."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.inference_times_ms: Deque[float] = deque(maxlen=window_size)
        self.ocr_times_ms: Deque[float] = deque(maxlen=window_size)
        self.currency_times_ms: Deque[float] = deque(maxlen=window_size)
        self.audio_latencies_ms: Deque[float] = deque(maxlen=window_size)
        self._last_frame_timestamp: float | None = None
        self.process = psutil.Process()

    def record_frame(self) -> float:
        """Record a processed frame timestamp and return current instantaneous FPS."""
        now = time.perf_counter()
        if self._last_frame_timestamp is not None:
            dt = now - self._last_frame_timestamp
            if dt > 0:
                self.frame_times.append(dt)
        self._last_frame_timestamp = now
        return self.get_fps()

    def record_inference(self, duration_ms: float, module: str = "object") -> None:
        """Record inference duration in milliseconds for a specific module."""
        if module == "object":
            self.inference_times_ms.append(duration_ms)
        elif module == "ocr":
            self.ocr_times_ms.append(duration_ms)
        elif module == "currency":
            self.currency_times_ms.append(duration_ms)

    def record_audio_latency(self, latency_ms: float) -> None:
        """Record end-to-end audio dispatch latency in milliseconds."""
        self.audio_latencies_ms.append(latency_ms)

    def get_fps(self) -> float:
        """Calculate moving-average FPS."""
        if not self.frame_times:
            return 0.0
        avg_dt = sum(self.frame_times) / len(self.frame_times)
        return round(1.0 / avg_dt, 2) if avg_dt > 0 else 0.0

    def get_avg_inference_ms(self, module: str = "object") -> float:
        """Return average inference latency in ms for the specified module."""
        target = {
            "object": self.inference_times_ms,
            "ocr": self.ocr_times_ms,
            "currency": self.currency_times_ms,
        }.get(module, self.inference_times_ms)
        if not target:
            return 0.0
        return round(sum(target) / len(target), 2)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect current CPU, RAM, and latency metrics."""
        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        proc_mem_mb = self.process.memory_info().rss / (1024 * 1024)
        avg_audio_ms = (
            round(sum(self.audio_latencies_ms) / len(self.audio_latencies_ms), 2)
            if self.audio_latencies_ms
            else 0.0
        )
        return {
            "fps": self.get_fps(),
            "object_inference_ms": self.get_avg_inference_ms("object"),
            "ocr_inference_ms": self.get_avg_inference_ms("ocr"),
            "currency_inference_ms": self.get_avg_inference_ms("currency"),
            "audio_latency_ms": avg_audio_ms,
            "cpu_percent": round(cpu_pct, 1),
            "system_ram_percent": round(vm.percent, 1),
            "process_ram_mb": round(proc_mem_mb, 1),
            "power_sensor_available": False,
        }
