"""
Litestar REST API for MedQCNN inference.

Endpoints:
  GET  /health  — Service health check
  GET  /info    — Model architecture details
  POST /predict — Run inference on a medical image

Usage:
    uv run python scripts/serve.py
    curl http://localhost:8000/health
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig

from medqcnn.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed

# --- Global model state ---
_model: HybridQCNN | None = None
_device: torch.device = torch.device("cpu")
_labels: list[str] = ["Benign", "Malignant"]


def load_model(
    checkpoint_path: str | None = None,
    n_qubits: int = DEMO_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    n_classes: int = 2,
) -> None:
    """Load the HybridQCNN model into global state."""
    global _model, _device

    set_seed()
    _device = get_device()

    _model = HybridQCNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        pretrained=True,
    ).to(_device)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(checkpoint["model_state_dict"])

    _model.eval()


@get("/health")
async def health_check() -> HealthResponse:
    """Service health check."""
    return HealthResponse()


@get("/info")
async def model_info() -> ModelInfoResponse:
    """Return model architecture details."""
    if _model is None:
        load_model()

    return ModelInfoResponse(
        n_qubits=_model.n_qubits,
        latent_dim=2**_model.n_qubits,
        n_ansatz_layers=_model.n_layers,
        n_classes=2,
        trainable_params=_model.count_trainable_params(),
        device=str(_device),
    )


@post("/predict")
async def predict(data: PredictionRequest) -> PredictionResponse:
    """Run inference on a base64-encoded medical image.

    Accepts a base64-encoded image, preprocesses it through the
    full hybrid pipeline, and returns:
    - Predicted class (0=Benign, 1=Malignant)
    - Confidence score
    - Class probabilities
    - Raw quantum expectation values
    """
    from litestar.exceptions import ClientException

    if _model is None:
        load_model()

    # Validate base64 payload size (max 10 MB encoded ~ 7.5 MB image)
    max_payload_bytes = 10 * 1024 * 1024
    if len(data.image_base64) > max_payload_bytes:
        raise ClientException(
            detail=f"Image payload too large. Max {max_payload_bytes} bytes.",
            status_code=413,
        )

    # Decode image from base64
    from PIL import Image
    from torchvision import transforms

    try:
        image_bytes = base64.b64decode(data.image_base64)
    except Exception:
        raise ClientException(
            detail="Invalid base64 encoding.",
            status_code=400,
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ClientException(
            detail="Cannot decode image. Supported formats: PNG, JPEG, BMP.",
            status_code=400,
        )

    # Validate image dimensions
    max_dim = 4096
    if image.width > max_dim or image.height > max_dim:
        raise ClientException(
            detail=(
                f"Image dimensions too large ({image.width}x{image.height}). "
                f"Max {max_dim}x{max_dim}."
            ),
            status_code=400,
        )

    image = image.convert("L")  # grayscale

    # Preprocess
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(_device)  # (1, 1, 224, 224)

    # Inference
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    # Get quantum expectation values
    q_values = None
    if hasattr(_model, "quantum_layer"):
        with torch.no_grad():
            features = _model.backbone(tensor)
            z = _model.projector(features)
            q_out = _model.quantum_layer(z)
            q_values = q_out[0].cpu().tolist()

    return PredictionResponse(
        prediction=pred,
        label=_labels[pred] if pred < len(_labels) else str(pred),
        confidence=round(confidence, 6),
        probabilities=probs[0].cpu().tolist() if data.return_probabilities else None,
        quantum_expectation_values=q_values,
    )


def create_app(checkpoint_path: str | None = None) -> Litestar:
    """Create and return the Litestar application."""
    load_model(checkpoint_path=checkpoint_path)

    return Litestar(
        route_handlers=[health_check, model_info, predict],
        cors_config=CORSConfig(allow_origins=["*"]),
    )
