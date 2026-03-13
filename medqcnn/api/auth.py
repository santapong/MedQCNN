"""
Authentication middleware for MedQCNN API.

Supports two auth modes:
  - Bearer JWT token: Authorization: Bearer <token>
  - API key header: X-API-Key: <key>

Public endpoints (/health, /schema) bypass auth.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.middleware import AbstractAuthenticationMiddleware, AuthenticationResult

from medqcnn.config.constants import JWT_ALGORITHM, JWT_EXPIRE_MINUTES

logger = logging.getLogger("medqcnn.auth")

# Public paths that skip authentication
PUBLIC_PATHS: set[str] = {"/health", "/schema", "/auth/token"}


def _get_jwt_secret() -> str:
    """Read JWT secret from environment."""
    secret = os.environ.get("JWT_SECRET_KEY", "")
    if not secret:
        logger.warning("JWT_SECRET_KEY not set — using insecure default for dev only")
        return "medqcnn-dev-secret-change-me"
    return secret


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash an API key for safe storage."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(32)


def create_jwt_token(
    subject: str,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT token."""
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=expires_minutes or JWT_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises on invalid/expired."""
    return jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])


def _validate_api_key(key: str) -> bool:
    """Check an API key against DB or environment fallback."""
    # Fast-path: check env var API key (single-key mode for edge deploy)
    env_key = os.environ.get("MEDQCNN_API_KEY", "")
    if env_key and secrets.compare_digest(key, env_key):
        return True

    # Check database for stored API keys
    try:
        from medqcnn.db.connection import get_session, init_db
        from medqcnn.db.crud import get_active_api_key_by_hash

        init_db()
        session = get_session()
        try:
            key_hash = hash_api_key(key)
            return get_active_api_key_by_hash(session, key_hash) is not None
        finally:
            session.close()
    except Exception:
        logger.debug(
            "DB API key lookup failed, falling back to env-only", exc_info=True
        )
        return False


class MedQCNNAuthMiddleware(AbstractAuthenticationMiddleware):
    """Litestar authentication middleware for JWT and API key auth."""

    async def authenticate_request(
        self, connection: ASGIConnection
    ) -> AuthenticationResult:
        # Skip auth for public endpoints
        if connection.scope["path"] in PUBLIC_PATHS:
            return AuthenticationResult(user={"role": "public"}, auth=None)

        # Check for disabled auth (dev mode)
        if os.environ.get("MEDQCNN_AUTH_DISABLED", "").lower() in ("1", "true"):
            return AuthenticationResult(user={"role": "dev"}, auth=None)

        # Try Bearer JWT
        auth_header = connection.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                payload = decode_jwt_token(token)
                return AuthenticationResult(
                    user={"sub": payload.get("sub", "unknown"), "role": "jwt"},
                    auth=payload,
                )
            except jwt.ExpiredSignatureError:
                raise NotAuthorizedException(detail="Token expired")
            except jwt.InvalidTokenError:
                raise NotAuthorizedException(detail="Invalid token")

        # Try X-API-Key header
        api_key = connection.headers.get("x-api-key", "")
        if api_key and _validate_api_key(api_key):
            return AuthenticationResult(
                user={"role": "api_key"},
                auth={"method": "api_key"},
            )

        raise NotAuthorizedException(detail="Missing or invalid authentication")
