"""Integration tests for the MedQCNN REST API endpoints."""

from __future__ import annotations

import base64
import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _disable_auth():
    """Disable auth for all integration tests."""
    os.environ["MEDQCNN_AUTH_DISABLED"] = "1"
    os.environ["JWT_SECRET_KEY"] = ""
    yield
    os.environ.pop("MEDQCNN_AUTH_DISABLED", None)
    os.environ.pop("JWT_SECRET_KEY", None)


@pytest.fixture(autouse=True)
def _use_sqlite_memory():
    """Use in-memory SQLite for tests."""
    os.environ["DATABASE_URL"] = "sqlite://"
    yield
    os.environ.pop("DATABASE_URL", None)

    from medqcnn.db.connection import reset_engine

    reset_engine()


@pytest.fixture()
def app():
    """Create a test app with a small 4-qubit model (no pretrained weights)."""
    from litestar import Litestar
    from litestar.config.cors import CORSConfig

    from medqcnn.api.auth import MedQCNNAuthMiddleware
    from medqcnn.api.model_service import model_service
    from medqcnn.api.rate_limit import RateLimitMiddleware
    from medqcnn.api.security import SecurityHeadersMiddleware
    from medqcnn.api.server import (
        activate_model,
        get_auth_token,
        get_metrics,
        get_prediction_endpoint,
        get_training_run_endpoint,
        health_check,
        list_benchmarks_endpoint,
        list_models,
        list_predictions_endpoint,
        list_training_runs_endpoint,
        model_info,
        predict,
        predict_batch,
        predict_dicom,
    )
    from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
    from medqcnn.db.connection import init_db
    from medqcnn.model.hybrid import HybridQCNN

    # Reset model service state
    model_service.model = None

    # Build model without pretrained weights (avoids network download)
    model_service.model = HybridQCNN(
        n_qubits=DEMO_QUBITS,
        n_layers=NUM_ANSATZ_LAYERS,
        n_classes=2,
        pretrained=False,
    )
    model_service.model.eval()

    init_db()

    application = Litestar(
        route_handlers=[
            health_check,
            model_info,
            predict,
            predict_batch,
            list_predictions_endpoint,
            get_prediction_endpoint,
            list_training_runs_endpoint,
            get_training_run_endpoint,
            list_benchmarks_endpoint,
            get_auth_token,
            list_models,
            activate_model,
            predict_dicom,
            get_metrics,
        ],
        middleware=[
            SecurityHeadersMiddleware,
            MedQCNNAuthMiddleware,
            RateLimitMiddleware,
        ],
        cors_config=CORSConfig(allow_origins=["http://localhost:3000"]),
    )
    return application


@pytest.fixture()
def client(app):
    """Create a Litestar test client."""
    from litestar.testing import TestClient

    with TestClient(app=app) as tc:
        yield tc


@pytest.fixture()
def sample_image_b64() -> str:
    """Generate a small valid grayscale PNG image as base64."""
    import io

    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "model_loaded" in data
        assert "db_connected" in data

    def test_health_includes_uptime(self, client):
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestInfoEndpoint:
    """Tests for GET /info."""

    def test_info_returns_model_details(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "n_qubits" in data
        assert "latent_dim" in data
        assert "trainable_params" in data
        assert data["latent_dim"] == 2 ** data["n_qubits"]


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_valid_image(self, client, sample_image_b64):
        response = client.post("/predict", json={"image_base64": sample_image_b64})
        assert response.status_code == 201
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "confidence" in data
        assert data["label"] in ("Benign", "Malignant")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_invalid_base64(self, client):
        response = client.post("/predict", json={"image_base64": "not-valid-b64!!!"})
        assert response.status_code == 400

    def test_predict_returns_quantum_values(self, client, sample_image_b64):
        response = client.post("/predict", json={"image_base64": sample_image_b64})
        data = response.json()
        assert "quantum_expectation_values" in data
        if data["quantum_expectation_values"] is not None:
            assert isinstance(data["quantum_expectation_values"], list)


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_predict(self, client, sample_image_b64):
        response = client.post(
            "/predict/batch",
            json={"images": [sample_image_b64, sample_image_b64]},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["total"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2

    def test_batch_predict_empty(self, client):
        response = client.post("/predict/batch", json={"images": []})
        assert response.status_code == 400

    def test_batch_predict_with_invalid_image(self, client, sample_image_b64):
        response = client.post(
            "/predict/batch",
            json={"images": [sample_image_b64, "bad-data"]},
        )
        data = response.json()
        assert data["total"] == 2
        assert data["failed"] >= 1


class TestPredictionHistory:
    """Tests for GET /predictions."""

    def test_list_predictions_empty(self, client):
        response = client.get("/predictions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_predictions_after_predict(self, client, sample_image_b64):
        # Create a prediction first
        client.post("/predict", json={"image_base64": sample_image_b64})

        response = client.get("/predictions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    def test_list_predictions_pagination(self, client):
        response = client.get("/predictions?offset=0&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 0
        assert data["limit"] == 10


class TestTrainingRuns:
    """Tests for GET /training-runs."""

    def test_list_training_runs(self, client):
        response = client.get("/training-runs")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestBenchmarks:
    """Tests for GET /benchmarks."""

    def test_list_benchmarks(self, client):
        response = client.get("/benchmarks")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestMetrics:
    """Tests for GET /metrics."""

    def test_get_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
