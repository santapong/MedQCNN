"""Tests for authentication module."""

from __future__ import annotations

import os

import pytest


class TestAuth:
    """Tests for JWT and API key authentication."""

    def test_generate_api_key(self):
        from medqcnn.api.auth import generate_api_key

        key = generate_api_key()
        assert len(key) > 20
        # Should be URL-safe base64
        assert all(c.isalnum() or c in "-_" for c in key)

    def test_hash_api_key_deterministic(self):
        from medqcnn.api.auth import hash_api_key

        h1 = hash_api_key("test-key")
        h2 = hash_api_key("test-key")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_api_key_different_inputs(self):
        from medqcnn.api.auth import hash_api_key

        h1 = hash_api_key("key-a")
        h2 = hash_api_key("key-b")
        assert h1 != h2

    def test_create_and_decode_jwt(self):
        from medqcnn.api.auth import create_jwt_token, decode_jwt_token

        os.environ["JWT_SECRET_KEY"] = "test-secret-12345"
        try:
            token = create_jwt_token(subject="test-user")
            payload = decode_jwt_token(token)
            assert payload["sub"] == "test-user"
            assert "exp" in payload
            assert "iat" in payload
        finally:
            del os.environ["JWT_SECRET_KEY"]

    def test_jwt_expired_token(self):
        import jwt as pyjwt

        from medqcnn.api.auth import create_jwt_token, decode_jwt_token

        os.environ["JWT_SECRET_KEY"] = "test-secret-12345"
        try:
            token = create_jwt_token(subject="test-user", expires_minutes=-1)
            with pytest.raises(pyjwt.ExpiredSignatureError):
                decode_jwt_token(token)
        finally:
            del os.environ["JWT_SECRET_KEY"]

    def test_validate_api_key_env(self):
        from medqcnn.api.auth import _validate_api_key

        os.environ["MEDQCNN_API_KEY"] = "my-test-key"
        try:
            assert _validate_api_key("my-test-key") is True
            assert _validate_api_key("wrong-key") is False
        finally:
            del os.environ["MEDQCNN_API_KEY"]
