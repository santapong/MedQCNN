"""
Split-conformal prediction sets with abstention.

Implements Adaptive Prediction Sets (APS — Romano, Sesia, Candès 2020)
on top of softmax probabilities from a trained classifier. Given a
held-out calibration split, APS produces a set-valued prediction with a
finite-sample marginal coverage guarantee:

    P( y ∈ C(x) ) ≥ 1 − α   (under exchangeability)

For a clinical decision-support deployment the set serves two roles:

* Singleton sets (|C(x)| == 1) → confident, single-class diagnosis.
* Non-singleton sets → flag for human review (the ``abstained`` field
  on `PredictionResponse` becomes True). On a 2-class task this
  effectively means "the model declines to commit", which is the
  behaviour FDA / IMDRF Good Machine Learning Practice asks for.

The predictor is fitted once after training, persisted as a JSON
sidecar next to the checkpoint, and applied at inference with a few
numpy ops — no extra forward passes, no extra model state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ConformalPredictor:
    """Split-conformal predictor using Adaptive Prediction Sets.

    Attributes:
        alpha: Miscoverage rate; coverage target is ``1 - alpha``.
        qhat: Calibrated conformity quantile. ``None`` until
            :py:meth:`calibrate` has been called.
        n_calibration: Size of the calibration split used to fit ``qhat``.
        n_classes: Number of classes the predictor was calibrated for.
    """

    alpha: float = 0.1
    qhat: float | None = None
    n_calibration: int = 0
    n_classes: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")

    @property
    def is_fitted(self) -> bool:
        return self.qhat is not None

    def calibrate(
        self,
        probs: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
    ) -> float:
        """Fit ``qhat`` from a calibration split.

        Args:
            probs: (N, C) softmax probabilities (already temperature-scaled
                if a calibration step is used).
            labels: (N,) integer class labels.

        Returns:
            The fitted quantile ``qhat``.
        """
        probs_np = _as_numpy(probs).astype(np.float64)
        labels_np = _as_numpy(labels).astype(np.int64)

        if probs_np.ndim != 2:
            raise ValueError(f"probs must be 2D (N, C), got shape {probs_np.shape}")
        if labels_np.shape[0] != probs_np.shape[0]:
            raise ValueError(
                "probs and labels must have matching first dim, got "
                f"{probs_np.shape[0]} vs {labels_np.shape[0]}"
            )

        n, c = probs_np.shape
        scores = _aps_scores(probs_np, labels_np)

        # Finite-sample correction: take the ⌈(n+1)(1-α)⌉ / n quantile.
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        qhat = float(np.quantile(scores, level, method="higher"))

        self.qhat = qhat
        self.n_calibration = int(n)
        self.n_classes = int(c)
        return qhat

    def predict_set(
        self,
        probs: torch.Tensor | np.ndarray,
    ) -> list[list[int]]:
        """Return a prediction set per row of ``probs``.

        Each set contains all classes whose accumulated descending
        softmax mass crosses ``qhat`` (the standard APS construction
        without the randomised tie-break — deterministic for clinical
        auditability).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ConformalPredictor.calibrate() must be called before predict_set()."
            )
        probs_np = _as_numpy(probs).astype(np.float64)
        if probs_np.ndim == 1:
            probs_np = probs_np[None, :]
        return _aps_predict_sets(probs_np, qhat=self.qhat)

    def abstains(
        self,
        probs: torch.Tensor | np.ndarray,
    ) -> list[bool]:
        """True for each row whose prediction set is not a singleton."""
        sets = self.predict_set(probs)
        return [len(s) != 1 for s in sets]

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "qhat": self.qhat,
            "n_calibration": self.n_calibration,
            "n_classes": self.n_classes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConformalPredictor:
        return cls(
            alpha=float(d["alpha"]),
            qhat=None if d.get("qhat") is None else float(d["qhat"]),
            n_calibration=int(d.get("n_calibration", 0)),
            n_classes=int(d.get("n_classes", 0)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ConformalPredictor:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ── Core APS arithmetic (kept module-private) ───────────────────


def _aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample APS conformity score.

    Sort probs in *descending* order; the conformity score is the
    cumulative mass up to and including the true label's rank. Higher
    scores indicate the true label was harder to surface.
    """
    n, _ = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)

    rank_of_true = (order == labels[:, None]).argmax(axis=1)
    return cumulative[np.arange(n), rank_of_true]


def _aps_predict_sets(probs: np.ndarray, qhat: float) -> list[list[int]]:
    """Build a prediction set per row whose cumulative mass ≤ qhat."""
    n, c = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)

    sets: list[list[int]] = []
    for i in range(n):
        # Include the rank at which cumulative first crosses qhat so the
        # coverage property holds — the set must contain at least one
        # class even if the top-1 mass already exceeds qhat.
        cross = int(np.searchsorted(cumulative[i], qhat, side="left"))
        cutoff = min(cross + 1, c)
        cutoff = max(cutoff, 1)
        sets.append(sorted(int(j) for j in order[i, :cutoff]))
    return sets


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
