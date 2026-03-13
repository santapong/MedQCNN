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
import binascii
import hashlib
import io
import logging

from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig
from litestar.exceptions import ClientException
from litestar.params import Parameter
from sqlalchemy.exc import SQLAlchemyError

from medqcnn.api.model_service import model_service
from medqcnn.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BenchmarkListResponse,
    DetailedHealthResponse,
    DicomMetadata,
    DicomPredictionResponse,
    MetricsResponse,
    ModelInfoResponse,
    ModelListResponse,
    ModelVersionInfo,
    PaginatedPredictions,
    PredictionDetail,
    PredictionRequest,
    PredictionResponse,
    TokenRequest,
    TokenResponse,
    TrainingRunListResponse,
)
from medqcnn.config.constants import DEMO_QUBITS

logger = logging.getLogger("medqcnn.api")


# ── Health & Info ────────────────────────────────────────


@get("/health")
async def health_check() -> DetailedHealthResponse:
    """Enhanced health check with dependency statuses."""
    from sqlalchemy import text

    from medqcnn.api.monitoring import metrics
    from medqcnn.utils.device import get_memory_info

    # Check database connectivity
    db_ok = False
    try:
        from medqcnn.db.connection import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except SQLAlchemyError:
        pass

    mem_info = get_memory_info()

    return DetailedHealthResponse(
        status="ok" if model_service.model is not None else "degraded",
        uptime_seconds=metrics.uptime_seconds,
        db_connected=db_ok,
        model_loaded=model_service.model is not None,
        model_version="0.2.0",
        memory_usage_mb=round(mem_info.get("ram_used_gb", 0) * 1024, 1),
    )


@get("/metrics")
async def get_metrics() -> MetricsResponse:
    """Return API usage metrics."""
    from medqcnn.api.monitoring import metrics

    snap = metrics.snapshot()
    return MetricsResponse(**snap)


@get("/info")
async def model_info() -> ModelInfoResponse:
    """Return model architecture details."""
    model_service.ensure_loaded()

    return ModelInfoResponse(
        n_qubits=model_service.model.n_qubits,
        latent_dim=2**model_service.model.n_qubits,
        n_ansatz_layers=model_service.model.n_layers,
        n_classes=2,
        trainable_params=model_service.model.count_trainable_params(),
        device=str(model_service.device),
    )


# ── Single Prediction ───────────────────────────────────


@post("/predict")
async def predict(data: PredictionRequest) -> PredictionResponse:
    """Run inference on a base64-encoded medical image."""
    result = model_service.run_inference(data.image_base64)

    # Auto-store in database
    try:
        from medqcnn.db.connection import db_session
        from medqcnn.db.crud import create_prediction

        with db_session() as session:
            image_hash = hashlib.sha256(data.image_base64.encode()[:1024]).hexdigest()[
                :16
            ]
            create_prediction(
                session,
                prediction=result.prediction,
                label=result.label,
                confidence=result.confidence,
                probabilities=result.probabilities,
                quantum_expectation_values=result.quantum_expectation_values,
                image_hash=image_hash,
                n_qubits=model_service.model.n_qubits if model_service.model else DEMO_QUBITS,
            )
    except SQLAlchemyError:
        logger.warning("Failed to store prediction in database", exc_info=True)

    return result


# ── Batch Prediction ─────────────────────────────────────


@post("/predict/batch")
async def predict_batch(data: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run inference on multiple base64-encoded medical images."""
    if len(data.images) == 0:
        raise ClientException(detail="No images provided.", status_code=400)
    if len(data.images) > 100:
        raise ClientException(detail="Max 100 images per batch.", status_code=400)

    results: list[PredictionResponse] = []
    errors: list[str] = []

    for i, img_b64 in enumerate(data.images):
        try:
            result = model_service.run_inference(img_b64)
            results.append(result)
        except ClientException as e:
            errors.append(f"Image {i}: {e.detail}")
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
        from medqcnn.db.connection import db_session
        from medqcnn.db.crud import create_prediction

        with db_session() as session:
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
                    n_qubits=model_service.model.n_qubits if model_service.model else DEMO_QUBITS,
                )
    except SQLAlchemyError:
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


# ── DICOM Prediction ───────────────────────────────────


@post("/predict/dicom")
async def predict_dicom(
    data: dict,
) -> DicomPredictionResponse:
    """Run inference on a base64-encoded DICOM file with metadata extraction."""
    from medqcnn.data.dicom import anonymize, dicom_to_pil, extract_metadata, read_dicom

    file_base64 = data.get("file_base64", "")
    if not file_base64:
        raise ClientException(detail="Missing file_base64 field.", status_code=400)

    try:
        file_bytes = base64.b64decode(file_base64)
    except binascii.Error:
        raise ClientException(detail="Invalid base64 encoding.", status_code=400)

    try:
        ds = read_dicom(file_bytes)
    except (ValueError, OSError) as exc:
        raise ClientException(detail=f"Cannot parse DICOM file: {exc}", status_code=400)

    # Anonymize and extract metadata
    anonymize(ds)
    metadata = extract_metadata(ds)

    # Convert to PIL and run through standard inference
    try:
        pil_image = dicom_to_pil(ds)
    except (ValueError, OSError) as exc:
        raise ClientException(
            detail=f"Cannot extract pixel data from DICOM: {exc}", status_code=400
        )

    # Encode as base64 PNG for reuse of run_inference
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    result = model_service.run_inference(image_b64)

    return DicomPredictionResponse(
        prediction=result.prediction,
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        quantum_expectation_values=result.quantum_expectation_values,
        dicom_metadata=DicomMetadata(**metadata),
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
    from medqcnn.api.security import sanitize_filename_search
    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import list_predictions

    with db_session() as session:
        rows, total = list_predictions(
            session,
            offset=offset,
            limit=limit,
            label_filter=label,
            confidence_min=confidence_min,
            filename_search=sanitize_filename_search(filename),
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


@get("/predictions/{prediction_id:int}")
async def get_prediction_endpoint(prediction_id: int) -> PredictionDetail:
    """Get a single prediction by ID."""
    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import get_prediction

    with db_session() as session:
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


# ── Training Runs ────────────────────────────────────────


@get("/training-runs")
async def list_training_runs_endpoint(
    offset: int = Parameter(default=0, ge=0),
    limit: int = Parameter(default=50, ge=1, le=200),
) -> TrainingRunListResponse:
    """List training runs with metrics."""
    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import list_training_runs

    with db_session() as session:
        rows, total = list_training_runs(session, offset=offset, limit=limit)
        return TrainingRunListResponse(
            items=[r.to_dict() for r in rows],
            total=total,
            offset=offset,
            limit=limit,
        )


@get("/training-runs/{run_id:int}")
async def get_training_run_endpoint(run_id: int) -> dict:
    """Get a single training run by ID."""
    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import get_training_run

    with db_session() as session:
        row = get_training_run(session, run_id)
        if row is None:
            raise ClientException(detail="Training run not found.", status_code=404)
        return row.to_dict()


# ── Benchmarks ───────────────────────────────────────────


@get("/benchmarks")
async def list_benchmarks_endpoint(
    training_run_id: int | None = None,
    offset: int = Parameter(default=0, ge=0),
    limit: int = Parameter(default=100, ge=1, le=500),
) -> BenchmarkListResponse:
    """List benchmarks, optionally filtered by training run."""
    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import list_benchmarks

    with db_session() as session:
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


# ── Authentication ───────────────────────────────────────


@post("/auth/token")
async def get_auth_token(data: TokenRequest) -> TokenResponse:
    """Exchange an API key for a JWT token."""
    from litestar.exceptions import NotAuthorizedException

    from medqcnn.api.auth import _validate_api_key, create_jwt_token

    if not _validate_api_key(data.api_key):
        raise NotAuthorizedException(detail="Invalid API key")

    token = create_jwt_token(subject="api_key_user")
    return TokenResponse(access_token=token, token_type="bearer")


# ── Model Versioning ────────────────────────────────────

_registry = None


def _get_registry():
    """Lazy-init model registry."""
    global _registry
    if _registry is None:
        from medqcnn.model.registry import ModelRegistry

        _registry = ModelRegistry()
    return _registry


@get("/models")
async def list_models() -> ModelListResponse:
    """List all available model versions."""
    registry = _get_registry()
    versions = registry.list_versions()
    return ModelListResponse(
        versions=[
            ModelVersionInfo(
                version=v.version,
                path=v.path,
                size_mb=round(v.size_bytes / (1024 * 1024), 2),
            )
            for v in versions
        ],
        active_version=registry.active_version,
    )


@post("/models/{version:str}/activate")
async def activate_model(version: str) -> dict:
    """Set the active model version."""
    registry = _get_registry()
    try:
        registry.set_active_version(version)
    except ValueError as e:
        raise ClientException(detail=str(e), status_code=404)
    return {"status": "ok", "active_version": version}


# ── App Factory ──────────────────────────────────────────


def create_app(checkpoint_path: str | None = None) -> Litestar:
    """Create and return the Litestar application."""
    import os

    from medqcnn.api.auth import MedQCNNAuthMiddleware
    from medqcnn.api.rate_limit import RateLimitMiddleware
    from medqcnn.api.security import SecurityHeadersMiddleware
    from medqcnn.db.connection import init_db

    model_service.load(checkpoint_path=checkpoint_path)
    init_db()

    # CORS: configurable via env, defaults to localhost for dev safety
    cors_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    if cors_env:
        allowed_origins = cors_env.split(",")
    else:
        logger.warning(
            "CORS_ALLOWED_ORIGINS not set — defaulting to http://localhost:3000. "
            "Set CORS_ALLOWED_ORIGINS env var for production."
        )
        allowed_origins = ["http://localhost:3000"]

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
        cors_config=CORSConfig(allow_origins=allowed_origins),
    )
