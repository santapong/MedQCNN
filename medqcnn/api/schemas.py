"""
Pydantic request/response schemas for the MedQCNN API.
"""

from __future__ import annotations

from msgspec import Struct


class HealthResponse(Struct):
    """Health check response."""

    status: str = "ok"
    service: str = "medqcnn"
    version: str = "0.1.0"


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
