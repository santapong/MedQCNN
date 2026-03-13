"""
Request/response schemas for the MedQCNN API.
"""

from __future__ import annotations

from msgspec import Struct


class HealthResponse(Struct):
    """Health check response."""

    status: str = "ok"
    service: str = "medqcnn"
    version: str = "0.2.0"


class ModelInfoResponse(Struct):
    """Model architecture info."""

    n_qubits: int
    latent_dim: int
    n_ansatz_layers: int
    n_classes: int
    trainable_params: dict[str, int]
    device: str


class PredictionRequest(Struct):
    """Inference request — base64-encoded image."""

    image_base64: str
    return_probabilities: bool = True


class PredictionResponse(Struct):
    """Inference response with diagnosis."""

    prediction: int
    label: str
    confidence: float
    probabilities: list[float] | None = None
    quantum_expectation_values: list[float] | None = None


class DatasetInfo(Struct):
    """Available dataset information."""

    name: str
    description: str
    n_classes: int
    n_samples: dict[str, int]


# ── Batch Prediction ─────────────────────────────────────


class BatchPredictionRequest(Struct):
    """Batch inference request — array of base64-encoded images."""

    images: list[str]


class BatchPredictionResponse(Struct):
    """Batch inference response with per-image results and summary."""

    results: list[PredictionResponse]
    total: int
    successful: int
    failed: int
    summary: dict[str, int | float]
    errors: list[str] | None = None


# ── Prediction History ───────────────────────────────────


class PredictionDetail(Struct):
    """Stored prediction with metadata."""

    id: int
    image_filename: str
    image_hash: str | None
    prediction: int
    label: str
    confidence: float
    probabilities: list[float] | None
    quantum_expectation_values: list[float] | None
    model_version: str
    n_qubits: int
    created_at: str | None


class PaginatedPredictions(Struct):
    """Paginated list of predictions."""

    items: list[PredictionDetail]
    total: int
    offset: int
    limit: int


# ── Training Runs & Benchmarks ───────────────────────────


class TrainingRunListResponse(Struct):
    """Paginated list of training runs."""

    items: list[dict]
    total: int
    offset: int
    limit: int


class BenchmarkListResponse(Struct):
    """Paginated list of benchmark metrics."""

    items: list[dict]
    total: int
    offset: int
    limit: int
