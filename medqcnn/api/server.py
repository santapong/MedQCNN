"""
Litestar REST API for MedQCNN inference.

Endpoints:
  GET  /health            — Service health check
  GET  /info              — Model architecture details
  POST /predict           — Run inference on a medical image
  POST /predict/batch     — Run inference on multiple images
  GET  /predictions       — List prediction history (paginated)
  GET  /predictions/{id}  — Single prediction detail
  GET  /training-runs     — List training runs with metrics
  GET  /training-runs/{id} — Single training run detail
  GET  /benchmarks        — Aggregated benchmark data

Usage:
    uv run python scripts/serve.py
    curl http://localhost:8000/health
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from pathlib import Path

import torch
from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig
from litestar.params import Parameter

from medqcnn.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BenchmarkListResponse,
    HealthResponse,
    ModelInfoResponse,
    PaginatedPredictions,
    PredictionDetail,
    PredictionRequest,
    PredictionResponse,
    TrainingRunListResponse,
)
from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed

logger = logging.getLogger("medqcnn.api")

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


def _ensure_model() -> None:
    """Lazy-load model if not already loaded."""
    if _model is None:
        load_model()


def _run_inference(image_base64: str) -> PredictionResponse:
    """Shared inference logic for single-image prediction."""
    from litestar.exceptions import ClientException
    from PIL import Image
    from torchvision import transforms

    _ensure_model()

    # Validate base64 payload size (max 250 MB encoded)
    max_payload_bytes = 250 * 1024 * 1024
    if len(image_base64) > max_payload_bytes:
        raise ClientException(
            detail=f"Image payload too large. Max {max_payload_bytes} bytes.",
            status_code=413,
        )

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        raise ClientException(
            detail="Invalid base64 encoding.",
            status_code=400,
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ClientException(
            detail="Cannot decode image. Supported formats: PNG, JPEG, BMP, TIFF.",
            status_code=400,
        )

    max_dim = 4096
    if image.width > max_dim or image.height > max_dim:
        raise ClientException(
            detail=(
                f"Image dimensions too large ({image.width}x{image.height}). "
                f"Max {max_dim}x{max_dim}."
            ),
            status_code=400,
        )

    image = image.convert("L")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

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
        probabilities=probs[0].cpu().tolist(),
        quantum_expectation_values=q_values,
    )


# ── Health & Info ────────────────────────────────────────


@get("/health")
async def health_check() -> HealthResponse:
    """Service health check."""
    return HealthResponse()


@get("/info")
async def model_info() -> ModelInfoResponse:
    """Return model architecture details."""
    _ensure_model()

    return ModelInfoResponse(
        n_qubits=_model.n_qubits,
        latent_dim=2**_model.n_qubits,
        n_ansatz_layers=_model.n_layers,
        n_classes=2,
        trainable_params=_model.count_trainable_params(),
        device=str(_device),
    )


# ── Single Prediction ───────────────────────────────────


@post("/predict")
async def predict(data: PredictionRequest) -> PredictionResponse:
    """Run inference on a base64-encoded medical image."""
    result = _run_inference(data.image_base64)

    # Auto-store in database
    try:
        from medqcnn.db.connection import get_session, init_db
        from medqcnn.db.crud import create_prediction

        init_db()
        session = get_session()
        try:
            image_hash = hashlib.sha256(
                data.image_base64.encode()[:1024]
            ).hexdigest()[:16]
            create_prediction(
                session,
                prediction=result.prediction,
                label=result.label,
                confidence=result.confidence,
                probabilities=result.probabilities,
                quantum_expectation_values=result.quantum_expectation_values,
                image_hash=image_hash,
                n_qubits=_model.n_qubits if _model else DEMO_QUBITS,
            )
        finally:
            session.close()
    except Exception:
        logger.warning("Failed to store prediction in database", exc_info=True)

    return result


# ── Batch Prediction ─────────────────────────────────────


@post("/predict/batch")
async def predict_batch(data: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run inference on multiple base64-encoded medical images."""
    from litestar.exceptions import ClientException

    if len(data.images) == 0:
        raise ClientException(detail="No images provided.", status_code=400)
    if len(data.images) > 100:
        raise ClientException(detail="Max 100 images per batch.", status_code=400)

    results: list[PredictionResponse] = []
    errors: list[str] = []

    for i, img_b64 in enumerate(data.images):
        try:
            result = _run_inference(img_b64)
            results.append(result)
        except Exception as e:
            errors.append(f"Image {i}: {e}")
            results.append(
                PredictionResponse(
                    prediction=-1,
                    label="error",
                    confidence=0.0,
                    probabilities=None,
                    quantum_expectation_values=None,
                )
            )

    # Auto-store successful predictions
    try:
        from medqcnn.db.connection import get_session, init_db
        from medqcnn.db.crud import create_prediction

        init_db()
        session = get_session()
        try:
            for i, result in enumerate(results):
                if result.label == "error":
                    continue
                create_prediction(
                    session,
                    prediction=result.prediction,
                    label=result.label,
                    confidence=result.confidence,
                    probabilities=result.probabilities,
                    quantum_expectation_values=result.quantum_expectation_values,
                    image_filename=f"batch_{i}",
                    n_qubits=_model.n_qubits if _model else DEMO_QUBITS,
                )
        finally:
            session.close()
    except Exception:
        logger.warning("Failed to store batch predictions in database", exc_info=True)

    # Summary
    valid_results = [r for r in results if r.label != "error"]
    benign_count = sum(1 for r in valid_results if r.label == "Benign")
    malignant_count = sum(1 for r in valid_results if r.label == "Malignant")
    avg_confidence = (
        sum(r.confidence for r in valid_results) / len(valid_results)
        if valid_results
        else 0.0
    )

    return BatchPredictionResponse(
        results=results,
        total=len(data.images),
        successful=len(valid_results),
        failed=len(errors),
        summary={
            "benign": benign_count,
            "malignant": malignant_count,
            "avg_confidence": round(avg_confidence, 4),
        },
        errors=errors if errors else None,
    )


# ── Prediction History ───────────────────────────────────


@get("/predictions")
async def list_predictions_endpoint(
    offset: int = Parameter(default=0, ge=0),
    limit: int = Parameter(default=50, ge=1, le=200),
    label: str | None = None,
    confidence_min: float | None = None,
    filename: str | None = None,
) -> PaginatedPredictions:
    """List prediction history with filtering and pagination."""
    from medqcnn.db.connection import get_session, init_db
    from medqcnn.db.crud import list_predictions

    init_db()
    session = get_session()
    try:
        rows, total = list_predictions(
            session,
            offset=offset,
            limit=limit,
            label_filter=label,
            confidence_min=confidence_min,
            filename_search=filename,
        )
        return PaginatedPredictions(
            items=[
                PredictionDetail(
                    id=r.id,
                    image_filename=r.image_filename,
                    image_hash=r.image_hash,
                    prediction=r.prediction,
                    label=r.label,
                    confidence=r.confidence,
                    probabilities=r.probabilities,
                    quantum_expectation_values=r.quantum_expectation_values,
                    model_version=r.model_version,
                    n_qubits=r.n_qubits,
                    created_at=r.created_at.isoformat() if r.created_at else None,
                )
                for r in rows
            ],
            total=total,
            offset=offset,
            limit=limit,
        )
    finally:
        session.close()


@get("/predictions/{prediction_id:int}")
async def get_prediction_endpoint(prediction_id: int) -> PredictionDetail:
    """Get a single prediction by ID."""
    from litestar.exceptions import ClientException

    from medqcnn.db.connection import get_session, init_db
    from medqcnn.db.crud import get_prediction

    init_db()
    session = get_session()
    try:
        row = get_prediction(session, prediction_id)
        if row is None:
            raise ClientException(detail="Prediction not found.", status_code=404)
        return PredictionDetail(
            id=row.id,
            image_filename=row.image_filename,
            image_hash=row.image_hash,
            prediction=row.prediction,
            label=row.label,
            confidence=row.confidence,
            probabilities=row.probabilities,
            quantum_expectation_values=row.quantum_expectation_values,
            model_version=row.model_version,
            n_qubits=row.n_qubits,
            created_at=row.created_at.isoformat() if row.created_at else None,
        )
    finally:
        session.close()


# ── Training Runs ────────────────────────────────────────


@get("/training-runs")
async def list_training_runs_endpoint(
    offset: int = Parameter(default=0, ge=0),
    limit: int = Parameter(default=50, ge=1, le=200),
) -> TrainingRunListResponse:
    """List training runs with metrics."""
    from medqcnn.db.connection import get_session, init_db
    from medqcnn.db.crud import list_training_runs

    init_db()
    session = get_session()
    try:
        rows, total = list_training_runs(session, offset=offset, limit=limit)
        return TrainingRunListResponse(
            items=[r.to_dict() for r in rows],
            total=total,
            offset=offset,
            limit=limit,
        )
    finally:
        session.close()


@get("/training-runs/{run_id:int}")
async def get_training_run_endpoint(run_id: int) -> dict:
    """Get a single training run by ID."""
    from litestar.exceptions import ClientException

    from medqcnn.db.connection import get_session, init_db
    from medqcnn.db.crud import get_training_run

    init_db()
    session = get_session()
    try:
        row = get_training_run(session, run_id)
        if row is None:
            raise ClientException(detail="Training run not found.", status_code=404)
        return row.to_dict()
    finally:
        session.close()


# ── Benchmarks ───────────────────────────────────────────


@get("/benchmarks")
async def list_benchmarks_endpoint(
    training_run_id: int | None = None,
    offset: int = Parameter(default=0, ge=0),
    limit: int = Parameter(default=100, ge=1, le=500),
) -> BenchmarkListResponse:
    """List benchmarks, optionally filtered by training run."""
    from medqcnn.db.connection import get_session, init_db
    from medqcnn.db.crud import list_benchmarks

    init_db()
    session = get_session()
    try:
        rows, total = list_benchmarks(
            session,
            training_run_id=training_run_id,
            offset=offset,
            limit=limit,
        )
        return BenchmarkListResponse(
            items=[r.to_dict() for r in rows],
            total=total,
            offset=offset,
            limit=limit,
        )
    finally:
        session.close()


# ── App Factory ──────────────────────────────────────────


def create_app(checkpoint_path: str | None = None) -> Litestar:
    """Create and return the Litestar application."""
    from medqcnn.db.connection import init_db

    load_model(checkpoint_path=checkpoint_path)
    init_db()

    return Litestar(
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
        ],
        cors_config=CORSConfig(allow_origins=["*"]),
    )
