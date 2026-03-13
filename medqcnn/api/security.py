"""
Security headers and input sanitization for MedQCNN API.

Implements OWASP-recommended response headers and input validation.
"""

from __future__ import annotations

import re

from litestar.enums import ScopeType
from litestar.middleware import AbstractMiddleware
from litestar.types import Receive, Scope, Send

# OWASP-recommended security headers
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

# Max allowed lengths for query parameters
MAX_QUERY_STRING_LENGTH = 256
MAX_FILENAME_SEARCH_LENGTH = 128
MAX_PAGINATION_LIMIT = 200

# Pattern for safe filename search (alphanumeric, dots, hyphens, underscores)
SAFE_FILENAME_PATTERN = re.compile(r"^[\w.\-\s]+$")


def sanitize_filename_search(value: str | None) -> str | None:
    """Sanitize filename search parameter to prevent injection."""
    if value is None:
        return None
    value = value[:MAX_FILENAME_SEARCH_LENGTH]
    if not SAFE_FILENAME_PATTERN.match(value):
        # Strip any characters that aren't safe
        value = re.sub(r"[^\w.\-\s]", "", value)
    return value or None


class SecurityHeadersMiddleware(AbstractMiddleware):
    """Add OWASP security headers to all responses."""

    scopes = {ScopeType.HTTP}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                for name, value in SECURITY_HEADERS.items():
                    headers.append((name.lower().encode(), value.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
