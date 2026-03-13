"""
In-memory sliding window rate limiter for MedQCNN API.

Designed for edge deployment — no Redis or external dependencies.
Tracks request timestamps per client IP with automatic cleanup.
"""

from __future__ import annotations

import time
from collections import defaultdict

from litestar.enums import ScopeType
from litestar.middleware import AbstractMiddleware
from litestar.response import Response
from litestar.types import Receive, Scope, Send

from medqcnn.config.constants import RATE_LIMIT_BATCH, RATE_LIMIT_PREDICT

# Per-path rate limits (requests per minute)
RATE_LIMITS: dict[str, int] = {
    "/predict": RATE_LIMIT_PREDICT,
    "/predict/batch": RATE_LIMIT_BATCH,
}

# Sliding window storage: {path: {ip: [timestamps]}}
_windows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

WINDOW_SECONDS = 60


def _cleanup_old_entries(timestamps: list[float], now: float) -> list[float]:
    """Remove timestamps older than the window."""
    cutoff = now - WINDOW_SECONDS
    return [t for t in timestamps if t > cutoff]


def check_rate_limit(path: str, client_ip: str) -> tuple[bool, int]:
    """Check if request is within rate limit.

    Returns:
        (allowed, retry_after_seconds)
    """
    limit = RATE_LIMITS.get(path)
    if limit is None:
        return True, 0

    now = time.monotonic()
    timestamps = _cleanup_old_entries(_windows[path][client_ip], now)
    _windows[path][client_ip] = timestamps

    if len(timestamps) >= limit:
        oldest = timestamps[0]
        retry_after = int(WINDOW_SECONDS - (now - oldest)) + 1
        return False, max(retry_after, 1)

    timestamps.append(now)
    return True, 0


class RateLimitMiddleware(AbstractMiddleware):
    """Litestar middleware that enforces per-IP rate limits on predict endpoints."""

    scopes = {ScopeType.HTTP}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")

        if path not in RATE_LIMITS:
            await self.app(scope, receive, send)
            return

        # Extract client IP
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        allowed, retry_after = check_rate_limit(path, client_ip)

        if not allowed:
            response = Response(
                content={"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
