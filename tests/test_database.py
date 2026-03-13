"""Tests for the MedQCNN database layer."""

from __future__ import annotations

import os

import pytest

# Force SQLite for testing
os.environ["DATABASE_URL"] = "sqlite:///test_medqcnn.db"

from medqcnn.db.connection import get_session, init_db, reset_engine
from medqcnn.db.crud import (
    create_benchmark,
    create_prediction,
    create_training_run,
    get_prediction,
    list_benchmarks,
    list_predictions,
    list_training_runs,
)


@pytest.fixture(autouse=True)
def setup_db():
    """Create a fresh test database for each test."""
    reset_engine()
    os.environ["DATABASE_URL"] = "sqlite:///test_medqcnn.db"
    init_db()
    yield
    # Cleanup
    reset_engine()
    if os.path.exists("test_medqcnn.db"):
        os.remove("test_medqcnn.db")


class TestPredictions:
    def test_create_prediction(self):
        session = get_session()
        try:
            row = create_prediction(
                session,
                prediction=0,
                label="Benign",
                confidence=0.85,
                probabilities=[0.85, 0.15],
                quantum_expectation_values=[0.1, -0.2, 0.3, -0.4],
                image_filename="test.png",
                n_qubits=4,
            )
            assert row.id is not None
            assert row.label == "Benign"
            assert row.confidence == 0.85
            assert row.probabilities == [0.85, 0.15]
            assert row.quantum_expectation_values == [0.1, -0.2, 0.3, -0.4]
        finally:
            session.close()

    def test_list_predictions(self):
        session = get_session()
        try:
            create_prediction(
                session, prediction=0, label="Benign", confidence=0.9
            )
            create_prediction(
                session, prediction=1, label="Malignant", confidence=0.7
            )
            rows, total = list_predictions(session)
            assert total == 2
            assert len(rows) == 2
        finally:
            session.close()

    def test_filter_by_label(self):
        session = get_session()
        try:
            create_prediction(
                session, prediction=0, label="Benign", confidence=0.9
            )
            create_prediction(
                session, prediction=1, label="Malignant", confidence=0.7
            )
            rows, total = list_predictions(session, label_filter="Benign")
            assert total == 1
            assert rows[0].label == "Benign"
        finally:
            session.close()

    def test_get_prediction(self):
        session = get_session()
        try:
            created = create_prediction(
                session, prediction=0, label="Benign", confidence=0.9
            )
            fetched = get_prediction(session, created.id)
            assert fetched is not None
            assert fetched.id == created.id
        finally:
            session.close()


class TestTrainingRuns:
    def test_create_training_run(self):
        session = get_session()
        try:
            run = create_training_run(
                session,
                dataset="breastmnist",
                n_qubits=4,
                n_layers=4,
                epochs=10,
                learning_rate=0.001,
                batch_size=16,
                final_train_acc=0.85,
                final_val_acc=0.78,
                history={"train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4],
                         "train_acc": [0.7, 0.85], "val_acc": [0.65, 0.78]},
            )
            assert run.id is not None
            assert run.dataset == "breastmnist"
            assert run.history is not None
            assert len(run.history["train_loss"]) == 2
        finally:
            session.close()

    def test_list_training_runs(self):
        session = get_session()
        try:
            create_training_run(
                session, dataset="breastmnist", n_qubits=4, n_layers=4,
                epochs=10, learning_rate=0.001, batch_size=16,
            )
            rows, total = list_training_runs(session)
            assert total == 1
        finally:
            session.close()


class TestBenchmarks:
    def test_create_benchmark(self):
        session = get_session()
        try:
            run = create_training_run(
                session, dataset="breastmnist", n_qubits=4, n_layers=4,
                epochs=10, learning_rate=0.001, batch_size=16,
            )
            bm = create_benchmark(
                session,
                training_run_id=run.id,
                metric_name="test_accuracy",
                metric_value=0.82,
            )
            assert bm.id is not None
            assert bm.metric_name == "test_accuracy"
        finally:
            session.close()

    def test_list_benchmarks_by_run(self):
        session = get_session()
        try:
            run = create_training_run(
                session, dataset="breastmnist", n_qubits=4, n_layers=4,
                epochs=10, learning_rate=0.001, batch_size=16,
            )
            create_benchmark(
                session, training_run_id=run.id,
                metric_name="acc", metric_value=0.8,
            )
            create_benchmark(
                session, training_run_id=run.id,
                metric_name="f1", metric_value=0.75,
            )
            rows, total = list_benchmarks(session, training_run_id=run.id)
            assert total == 2
        finally:
            session.close()
