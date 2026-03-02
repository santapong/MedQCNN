"""
Training loop for the HybridQCNN model.

Implements the Adam-optimizer-based training loop with:
- Configurable epochs, learning rate, batch size
- Checkpoint saving/loading
- Training and validation metrics logging
- Progress tracking via Rich console
"""

from __future__ import annotations

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


class Trainer:
    """Training manager for the HybridQCNN model.

    Args:
        model: The HybridQCNN model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        learning_rate: Adam optimizer learning rate.
        checkpoint_dir: Directory for saving checkpoints.
        device: Compute device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        checkpoint_dir: str = CHECKPOINT_DIR,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )
        self.criterion = nn.CrossEntropyLoss()

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
            loss = self.criterion(logits, labels)
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
            loss = self.criterion(logits, labels)

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
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Args:
            epochs: Number of training epochs.
            save_every: Save checkpoint every N epochs.

        Returns:
            Training history dict.
        """
        print(f"Training on device: {self.device}")
        print(f"Trainable params: {self.model.count_trainable_params()}")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            elapsed = time.time() - start_time

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

        # Save final checkpoint
        self.save_checkpoint(epochs, filename="model_final.pt")
        return self.history

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
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint["epoch"]
