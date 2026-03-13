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


# ── Authentication ──────────────────────────────────────


class TokenRequest(Struct):
    """Exchange API key for JWT token."""

    api_key: str


class TokenResponse(Struct):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


# ── Model Versioning ────────────────────────────────────


class ModelVersionInfo(Struct):
    """Single model version metadata."""

    version: str
    path: str
    size_mb: float


class ModelListResponse(Struct):
    """List of available model versions."""

    versions: list[ModelVersionInfo]
    active_version: str | None


# ── DICOM ───────────────────────────────────────────────


class DicomMetadata(Struct):
    """Anonymized DICOM study metadata."""

    modality: str | None = None
    study_description: str | None = None
    body_part: str | None = None
    study_date: str | None = None
    institution: str | None = None
    rows: int | None = None
    columns: int | None = None


class DicomPredictionResponse(Struct):
    """Prediction result with DICOM metadata."""

    prediction: int
    label: str
    confidence: float
    probabilities: list[float] | None = None
    quantum_expectation_values: list[float] | None = None
    dicom_metadata: DicomMetadata | None = None


# ── Monitoring ──────────────────────────────────────────


class DetailedHealthResponse(Struct):
    """Enhanced health check with dependency statuses."""

    status: str = "ok"
    service: str = "medqcnn"
    version: str = "0.2.0"
    uptime_seconds: float = 0.0
    db_connected: bool = False
    model_loaded: bool = False
    model_version: str | None = None
    memory_usage_mb: float = 0.0


class MetricsResponse(Struct):
    """API usage metrics."""

    total_requests: int = 0
    total_predictions: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0
