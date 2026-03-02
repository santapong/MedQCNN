"""
Evaluation metrics for model benchmarking.

Provides AUC-ROC, accuracy, F1 score, and confusion matrix
computation for medical classification tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy in [0, 1].
    """
    return float(np.mean(y_true == y_pred))


def compute_auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted probabilities for the positive class.

    Returns:
        AUC-ROC score in [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """Compute F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method ('binary', 'micro', 'macro', 'weighted').

    Returns:
        F1 score.
    """
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average=average))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as a 2D array.
    """
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred)
