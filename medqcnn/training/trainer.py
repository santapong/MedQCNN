"""
Training loop for the HybridQCNN model.

Implements the Adam-optimizer-based training loop with:
- Configurable epochs, learning rate, batch size
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping with best checkpoint tracking
- Checkpoint saving/loading with epoch offset for resume
- Training and validation metrics logging
- Progress tracking via Rich console
- Auto-stores training run metadata in database on completion
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from medqcnn.config.constants import (
    CHECKPOINT_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
)
from medqcnn.training.loss import HybridLoss

logger = logging.getLogger("medqcnn.training")


class Trainer:
    """Training manager for the HybridQCNN model.

    Args:
        model: The HybridQCNN model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        learning_rate: Adam optimizer learning rate.
        checkpoint_dir: Directory for saving checkpoints.
        device: Compute device ('cpu' or 'cuda').
        dataset_name: Name of the dataset being trained on.
        n_qubits: Number of qubits in the quantum layer.
        n_layers: Number of ansatz layers.
        batch_size: Training batch size.
        labels: Human-readable class names (saved in checkpoints for inference).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        checkpoint_dir: str = CHECKPOINT_DIR,
        device: str = "cpu",
        dataset_name: str = "breastmnist",
        n_qubits: int = 8,
        n_layers: int = 4,
        batch_size: int = 16,
        labels: list[str] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training config metadata
        self.dataset_name = dataset_name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.labels = labels

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )
        self.criterion = HybridLoss(l2_lambda=1e-4)

        # LR scheduler: reduce on validation loss plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self) -> tuple[float, float]:
        """Run a single training epoch.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()

            self.optimizer.zero_grad()
            logits = self.model(images)

            # Gather quantum parameters for L2 regularization
            quantum_params = None
            if hasattr(self.model, "quantum_layer"):
                quantum_params = torch.cat(
                    [p.flatten() for p in self.model.quantum_layer.parameters()
                     if p.requires_grad]
                )

            loss = self.criterion(logits, labels, quantum_params=quantum_params)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Run validation.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device).squeeze()

            logits = self.model(images)
            # Validation loss uses CE only (no regularization)
            loss = self.criterion.ce_loss(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def fit(
        self,
        epochs: int = DEFAULT_EPOCHS,
        save_every: int = 10,
        patience: int = 10,
        resume_from: int = 0,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            epochs: Number of training epochs.
            save_every: Save checkpoint every N epochs.
            patience: Early stopping patience (epochs without val loss improvement).
                Set to 0 to disable early stopping.
            resume_from: Starting epoch offset (for resumed training).

        Returns:
            Training history dict.
        """
        print(f"Training on device: {self.device}")
        print(f"Trainable params: {self.model.count_trainable_params()}")

        if resume_from > 0:
            print(f"Resuming from epoch {resume_from}")

        start_time = time.time()
        best_val_loss = float("inf")
        patience_counter = 0
        epochs_completed = resume_from

        for epoch in range(resume_from + 1, resume_from + epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Step LR scheduler on validation loss
            self.scheduler.step(val_loss)

            elapsed = time.time() - epoch_start

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{resume_from + epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, filename="model_best.pt")
            else:
                patience_counter += 1

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

            epochs_completed = epoch

            # Early stopping
            if patience > 0 and patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no val loss improvement for {patience} epochs)"
                )
                break

        # Save final checkpoint
        self.save_checkpoint(epochs_completed, filename="model_final.pt")

        total_duration = time.time() - start_time

        # Auto-store training run in database
        self._store_training_run(epochs_completed, total_duration)

        return self.history

    def _store_training_run(self, epochs: int, duration_seconds: float) -> None:
        """Persist training run metadata to the database."""
        try:
            from medqcnn.db.connection import db_session
            from medqcnn.db.crud import create_benchmark, create_training_run

            with db_session() as session:
                final_train_acc = (
                    self.history["train_acc"][-1] if self.history["train_acc"] else None
                )
                final_val_acc = (
                    self.history["val_acc"][-1] if self.history["val_acc"] else None
                )

                run = create_training_run(
                    session,
                    dataset=self.dataset_name,
                    n_qubits=self.n_qubits,
                    n_layers=self.n_layers,
                    epochs=epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    final_train_acc=final_train_acc,
                    final_val_acc=final_val_acc,
                    duration_seconds=duration_seconds,
                    checkpoint_path=str(self.checkpoint_dir / "model_final.pt"),
                    history=self.history,
                )

                # Store key metrics as benchmarks
                param_counts = self.model.count_trainable_params()
                for name, count in param_counts.items():
                    create_benchmark(
                        session,
                        training_run_id=run.id,
                        metric_name=f"params_{name}",
                        metric_value=float(count),
                    )
                if final_train_acc is not None:
                    create_benchmark(
                        session,
                        training_run_id=run.id,
                        metric_name="final_train_acc",
                        metric_value=final_train_acc,
                    )
                if final_val_acc is not None:
                    create_benchmark(
                        session,
                        training_run_id=run.id,
                        metric_name="final_val_acc",
                        metric_value=final_val_acc,
                    )

                logger.info("Stored training run #%d in database", run.id)
        except Exception:
            logger.warning("Failed to store training run in database", exc_info=True)

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> None:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "history": self.history,
                "labels": self.labels,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.labels = checkpoint.get("labels", self.labels)
        return checkpoint["epoch"]
