"""
Post-hoc calibration for the HybridQCNN classifier.

Implements temperature scaling (Guo et al., 2017) — a single scalar
T > 0 that rescales logits as ``logits / T`` before softmax. Fitted by
minimising negative log-likelihood on a held-out calibration split with
L-BFGS. Cheap (one parameter), preserves the argmax, and consistently
reduces Expected Calibration Error for over-confident classifiers — a
known failure mode of CE-trained networks with a small classification
head, which is exactly the regime of `HybridQCNN.classifier`.

The module also exposes ECE and a binned reliability curve so the
frontend / paper can plot pre- and post-calibration diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by a positive temperature."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    return logits / temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Fit a single scalar temperature via L-BFGS on NLL.

    Args:
        logits: (N, C) raw classifier logits on the calibration split.
        labels: (N,) integer class labels.
        max_iter: L-BFGS iteration cap.
        lr: L-BFGS learning rate.

    Returns:
        Fitted temperature as a Python float, clamped to ``[1e-3, 1e3]``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N, C), got shape {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"labels must be 1D with N={logits.shape[0]}, got {tuple(labels.shape)}"
        )

    logits = logits.detach().float()
    labels = labels.detach().long()

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_t.exp()
        loss = functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    return max(1e-3, min(1e3, temperature))


def compute_ece(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    Args:
        probs: (N, C) softmax probabilities.
        labels: (N,) integer ground-truth labels.
        n_bins: number of equal-width bins on [0, 1].

    Returns:
        ECE in [0, 1].
    """
    probs_np = _as_numpy(probs)
    labels_np = _as_numpy(labels).astype(np.int64)
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    accuracies = (predictions == labels_np).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(confidences))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_curve(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    n_bins: int = 15,
) -> list[ReliabilityBin]:
    """Per-bin (confidence, accuracy, count) tuples for plotting."""
    probs_np = _as_numpy(probs)
    labels_np = _as_numpy(labels).astype(np.int64)
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    accuracies = (predictions == labels_np).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                ReliabilityBin(
                    lower=lo, upper=hi, count=0, confidence=0.0, accuracy=0.0
                )
            )
            continue
        bins.append(
            ReliabilityBin(
                lower=lo,
                upper=hi,
                count=count,
                confidence=float(confidences[mask].mean()),
                accuracy=float(accuracies[mask].mean()),
            )
        )
    return bins


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
