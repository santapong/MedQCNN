"""Tests for monitoring module."""

from __future__ import annotations

from medqcnn.api.monitoring import APIMetrics


class TestAPIMetrics:
    """Tests for in-memory API metrics tracking."""

    def test_initial_state(self):
        m = APIMetrics()
        snap = m.snapshot()
        assert snap["total_requests"] == 0
        assert snap["total_predictions"] == 0
        assert snap["error_count"] == 0
        assert snap["uptime_seconds"] >= 0

    def test_record_request(self):
        m = APIMetrics()
        m.record_request()
        m.record_request()
        assert m.snapshot()["total_requests"] == 2

    def test_record_prediction_with_latency(self):
        m = APIMetrics()
        m.record_prediction(100.0)
        m.record_prediction(200.0)
        snap = m.snapshot()
        assert snap["total_predictions"] == 2
        assert snap["avg_latency_ms"] == 150.0

    def test_record_error(self):
        m = APIMetrics()
        m.record_error()
        assert m.snapshot()["error_count"] == 1

    def test_uptime_increases(self):
        import time

        m = APIMetrics()
        time.sleep(0.05)
        assert m.uptime_seconds > 0
