"""
In-memory metrics and uptime tracking for MedQCNN API.

Lightweight monitoring designed for edge deployment — no Prometheus
or external dependencies. Metrics reset on server restart.
"""

from __future__ import annotations

import threading
import time


class APIMetrics:
    """Thread-safe in-memory API metrics tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._request_count = 0
        self._prediction_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    @property
    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._start_time, 2)

    def record_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def record_prediction(self, latency_ms: float) -> None:
        with self._lock:
            self._prediction_count += 1
            self._total_latency_ms += latency_ms

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._prediction_count
                if self._prediction_count > 0
                else 0.0
            )
            return {
                "total_requests": self._request_count,
                "total_predictions": self._prediction_count,
                "avg_latency_ms": round(avg_latency, 2),
                "error_count": self._error_count,
                "uptime_seconds": self.uptime_seconds,
            }


# Global singleton
metrics = APIMetrics()
