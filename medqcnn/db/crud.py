"""
CRUD operations for MedQCNN database models.
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from medqcnn.db.models import APIKey, Benchmark, Prediction, TrainingRun

# ── Predictions ──────────────────────────────────────────


def create_prediction(
    session: Session,
    *,
    prediction: int,
    label: str,
    confidence: float,
    probabilities: list[float] | None = None,
    quantum_expectation_values: list[float] | None = None,
    image_filename: str = "unknown",
    image_hash: str | None = None,
    model_version: str = "0.1.0",
    n_qubits: int = 4,
) -> Prediction:
    """Insert a new prediction row and return it."""
    row = Prediction(
        image_filename=image_filename,
        image_hash=image_hash,
        prediction=prediction,
        label=label,
        confidence=confidence,
        probabilities_json=json.dumps(probabilities) if probabilities else None,
        quantum_values_json=(
            json.dumps(quantum_expectation_values)
            if quantum_expectation_values
            else None
        ),
        model_version=model_version,
        n_qubits=n_qubits,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_predictions(
    session: Session,
    *,
    offset: int = 0,
    limit: int = 50,
    label_filter: str | None = None,
    confidence_min: float | None = None,
    filename_search: str | None = None,
) -> tuple[list[Prediction], int]:
    """Return paginated predictions and total count."""
    q = session.query(Prediction)
    if label_filter:
        q = q.filter(Prediction.label == label_filter)
    if confidence_min is not None:
        q = q.filter(Prediction.confidence >= confidence_min)
    if filename_search:
        # Escape SQL LIKE wildcards to prevent injection
        escaped = filename_search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        q = q.filter(Prediction.image_filename.ilike(f"%{escaped}%", escape="\\"))
    total = q.count()
    rows = q.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()
    return rows, total


def get_prediction(session: Session, prediction_id: int) -> Prediction | None:
    """Fetch a single prediction by ID."""
    return session.get(Prediction, prediction_id)


# ── Training Runs ────────────────────────────────────────


def create_training_run(
    session: Session,
    *,
    dataset: str,
    n_qubits: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    final_train_acc: float | None = None,
    final_val_acc: float | None = None,
    final_test_acc: float | None = None,
    auc_roc: float | None = None,
    f1: float | None = None,
    duration_seconds: float | None = None,
    checkpoint_path: str | None = None,
    history: dict[str, list[float]] | None = None,
) -> TrainingRun:
    """Insert a new training run."""
    row = TrainingRun(
        dataset=dataset,
        n_qubits=n_qubits,
        n_layers=n_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        final_train_acc=final_train_acc,
        final_val_acc=final_val_acc,
        final_test_acc=final_test_acc,
        auc_roc=auc_roc,
        f1=f1,
        duration_seconds=duration_seconds,
        checkpoint_path=checkpoint_path,
        history_json=json.dumps(history) if history else None,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_training_runs(
    session: Session,
    *,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[TrainingRun], int]:
    """Return paginated training runs and total count."""
    q = session.query(TrainingRun)
    total = q.count()
    rows = q.order_by(TrainingRun.created_at.desc()).offset(offset).limit(limit).all()
    return rows, total


def get_training_run(session: Session, run_id: int) -> TrainingRun | None:
    """Fetch a single training run by ID."""
    return session.get(TrainingRun, run_id)


# ── Benchmarks ───────────────────────────────────────────


def create_benchmark(
    session: Session,
    *,
    training_run_id: int,
    metric_name: str,
    metric_value: float,
) -> Benchmark:
    """Insert a benchmark metric."""
    row = Benchmark(
        training_run_id=training_run_id,
        metric_name=metric_name,
        metric_value=metric_value,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_benchmarks(
    session: Session,
    *,
    training_run_id: int | None = None,
    offset: int = 0,
    limit: int = 100,
) -> tuple[list[Benchmark], int]:
    """Return paginated benchmarks, optionally filtered by training run."""
    q = session.query(Benchmark)
    if training_run_id is not None:
        q = q.filter(Benchmark.training_run_id == training_run_id)
    total = q.count()
    rows = q.order_by(Benchmark.created_at.desc()).offset(offset).limit(limit).all()
    return rows, total


# ── API Keys ────────────────────────────────────────────


def create_api_key(
    session: Session,
    *,
    name: str,
    key_hash: str,
) -> APIKey:
    """Insert a new API key (stores hash only)."""
    row = APIKey(name=name, key_hash=key_hash, is_active=True)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def get_active_api_key_by_hash(session: Session, key_hash: str) -> APIKey | None:
    """Look up an active API key by its hash."""
    return (
        session.query(APIKey)
        .filter(APIKey.key_hash == key_hash, APIKey.is_active.is_(True))
        .first()
    )
