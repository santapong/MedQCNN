"""
SQLAlchemy ORM models for MedQCNN persistent storage.

Tables:
  - predictions: Every inference result from /predict and /predict/batch
  - training_runs: Metadata from completed training sessions
  - benchmarks: Per-metric rows linked to training runs
"""

from __future__ import annotations

import datetime
import json

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all models."""


class Prediction(Base):
    """Stored inference result."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_filename: Mapped[str] = mapped_column(String(512), default="unknown")
    image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    quantum_values_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(32), default="0.1.0")
    n_qubits: Mapped[int] = mapped_column(Integer, default=4)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # --- helpers ---
    @property
    def probabilities(self) -> list[float] | None:
        if self.probabilities_json is None:
            return None
        return json.loads(self.probabilities_json)

    @property
    def quantum_expectation_values(self) -> list[float] | None:
        if self.quantum_values_json is None:
            return None
        return json.loads(self.quantum_values_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "image_hash": self.image_hash,
            "prediction": self.prediction,
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "quantum_expectation_values": self.quantum_expectation_values,
            "model_version": self.model_version,
            "n_qubits": self.n_qubits,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingRun(Base):
    """Metadata for a completed training session."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    n_qubits: Mapped[int] = mapped_column(Integer, nullable=False)
    n_layers: Mapped[int] = mapped_column(Integer, nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, nullable=False)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    final_train_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_val_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_test_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    auc_roc: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    history_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    benchmarks: Mapped[list[Benchmark]] = relationship(
        "Benchmark", back_populates="training_run", cascade="all, delete-orphan"
    )

    @property
    def history(self) -> dict[str, list[float]] | None:
        if self.history_json is None:
            return None
        return json.loads(self.history_json)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "final_train_acc": self.final_train_acc,
            "final_val_acc": self.final_val_acc,
            "final_test_acc": self.final_test_acc,
            "auc_roc": self.auc_roc,
            "f1": self.f1,
            "duration_seconds": self.duration_seconds,
            "checkpoint_path": self.checkpoint_path,
            "history": self.history,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
        }


class Benchmark(Base):
    """Individual metric value associated with a training run."""

    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("training_runs.id"), nullable=False
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    training_run: Mapped[TrainingRun] = relationship(
        "TrainingRun", back_populates="benchmarks"
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "training_run_id": self.training_run_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class APIKey(Base):
    """Stored API key for authentication."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
