"""
Training visualization utilities.

Generates publication-quality plots for training metrics:
- Loss and accuracy curves
- ROC curves with AUC score
- Confusion matrix heatmaps

All plots saved to docs/ by default for notebook embedding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = "docs/training_history.png",
) -> None:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss',
            'train_acc', 'val_acc'.
        save_path: Path to save the figure. None to skip saving.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    if history.get("val_loss") and any(v > 0 for v in history["val_loss"]):
        axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    if history.get("val_acc") and any(v > 0 for v in history["val_acc"]):
        axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: str | Path | None = "docs/roc_curve.png",
) -> float:
    """Plot ROC curve and return AUC score.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted probabilities for the positive class.
        save_path: Path to save the figure.

    Returns:
        AUC-ROC score.
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = "docs/confusion_matrix.png",
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional class label names.
        save_path: Path to save the figure.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )

    # Write values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
