"""Tests for rate limiting module."""

from __future__ import annotations

from medqcnn.api.rate_limit import _windows, check_rate_limit


class TestRateLimit:
    """Tests for sliding window rate limiter."""

    def setup_method(self):
        """Clear rate limit state between tests."""
        _windows.clear()

    def test_allows_within_limit(self):
        for _ in range(5):
            allowed, _ = check_rate_limit("/predict", "127.0.0.1")
            assert allowed is True

    def test_blocks_over_limit(self):
        # Exhaust the limit
        from medqcnn.api.rate_limit import RATE_LIMITS

        limit = RATE_LIMITS["/predict"]
        for _ in range(limit):
            check_rate_limit("/predict", "127.0.0.1")

        # Next request should be blocked
        allowed, retry_after = check_rate_limit("/predict", "127.0.0.1")
        assert allowed is False
        assert retry_after > 0

    def test_different_ips_independent(self):
        from medqcnn.api.rate_limit import RATE_LIMITS

        limit = RATE_LIMITS["/predict"]
        # Exhaust limit for IP 1
        for _ in range(limit):
            check_rate_limit("/predict", "10.0.0.1")

        # IP 2 should still be allowed
        allowed, _ = check_rate_limit("/predict", "10.0.0.2")
        assert allowed is True

    def test_non_limited_path_always_allowed(self):
        for _ in range(100):
            allowed, _ = check_rate_limit("/health", "127.0.0.1")
            assert allowed is True

    def test_batch_has_lower_limit(self):
        from medqcnn.api.rate_limit import RATE_LIMITS

        predict_limit = RATE_LIMITS["/predict"]
        batch_limit = RATE_LIMITS["/predict/batch"]
        assert batch_limit < predict_limit
