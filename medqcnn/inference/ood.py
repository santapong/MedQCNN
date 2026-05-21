"""
Out-of-distribution gate for HybridQCNN.

Fits a multivariate Gaussian to the frozen ResNet backbone features of
the training set, then at inference scores a new sample by its negative
log-likelihood under that Gaussian. Samples above a percentile-based
threshold (fit on a held-out validation split) are flagged as OOD —
the API surface ORs this flag with the conformal abstention signal so
``PredictionResponse.abstained`` becomes a single "do not act on this"
gate for the clinician-in-the-loop UI.

Why a Gaussian and not a normalizing flow?

* Backbone features sit in R^{d} with d=512 for ResNet-18. A diagonal-
  shrinkage Gaussian fits in O(d²) memory and one pass, and matches the
  Mahalanobis-distance OOD baselines from Lee et al., 2018 — strong
  numbers for the cost.
* A flow gives a tighter density model but requires a fitting loop and
  extra dependencies. We leave it as a follow-up; the on-disk schema
  here is intentionally generic (``method`` field) so a flow-based
  detector can be swapped in without changing the consumer.

The fitted detector is persisted as a JSON sidecar
``<ckpt>.ood.json`` next to the checkpoint and auto-loaded by
:class:`medqcnn.api.model_service.ModelService`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class OODDetector:
    """Gaussian-density OOD detector on backbone features.

    Attributes:
        method: Identifier kept in the sidecar so consumers can detect
            future-format swaps. Always ``"gaussian"`` for this class.
        feature_dim: Dimensionality ``d`` of the backbone feature vector.
        mean: Per-dim mean vector (shape ``(d,)``).
        cov_chol: Cholesky factor of the regularised covariance
            (shape ``(d, d)``, lower-triangular). Stored rather than the
            inverse so log-det is cheap.
        log_det: ``2 * sum(log(diag(cov_chol)))`` — pre-cached.
        regularization: Diagonal shrinkage ``λ`` added to the covariance
            (``cov + λ·I``) for invertibility.
        threshold: Negative log-likelihood threshold beyond which a
            sample is flagged OOD. ``None`` until
            :py:meth:`calibrate_threshold` runs.
        percentile: Percentile of in-distribution val NLL used to set
            ``threshold``.
        n_train: Size of the training-feature matrix used to fit
            ``mean`` and ``cov_chol``.
    """

    method: str = "gaussian"
    feature_dim: int = 0
    mean: np.ndarray | None = None
    cov_chol: np.ndarray | None = None
    log_det: float | None = None
    regularization: float = 1e-3
    threshold: float | None = None
    percentile: float = 95.0
    n_train: int = 0

    # ── Fitting ──────────────────────────────────────────────────

    def fit(
        self,
        features: torch.Tensor | np.ndarray,
        regularization: float = 1e-3,
    ) -> None:
        """Fit the Gaussian on a stack of in-distribution features."""
        feats = _as_numpy(features).astype(np.float64)
        if feats.ndim != 2:
            raise ValueError(
                f"features must be 2D (N, d); got shape {feats.shape}"
            )

        n, d = feats.shape
        mu = feats.mean(axis=0)
        centered = feats - mu
        cov = (centered.T @ centered) / max(1, n - 1)
        cov = cov + regularization * np.eye(d)

        cov_chol = np.linalg.cholesky(cov)
        log_det = float(2.0 * np.sum(np.log(np.diag(cov_chol))))

        self.method = "gaussian"
        self.feature_dim = int(d)
        self.mean = mu
        self.cov_chol = cov_chol
        self.log_det = log_det
        self.regularization = float(regularization)
        self.n_train = int(n)

    def calibrate_threshold(
        self,
        val_features: torch.Tensor | np.ndarray,
        percentile: float = 95.0,
    ) -> float:
        """Set ``self.threshold`` from a percentile of in-dist val NLL.

        Args:
            val_features: (N, d) features from the in-distribution val
                split. Should *not* overlap with the fit set.
            percentile: Percentile of NLL beyond which we flag OOD.
                95 means roughly the worst 5% of in-dist samples will be
                flagged — a typical operating point.

        Returns:
            The fitted threshold.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before calibrate_threshold().")
        scores = self.score(val_features)
        threshold = float(np.percentile(scores, percentile))
        self.threshold = threshold
        self.percentile = float(percentile)
        return threshold

    # ── Inference ────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return (
            self.mean is not None
            and self.cov_chol is not None
            and self.log_det is not None
        )

    def score(
        self,
        features: torch.Tensor | np.ndarray,
    ) -> np.ndarray:
        """Per-sample negative log-likelihood under the fitted Gaussian.

        Higher score → less in-distribution. Returns shape ``(N,)`` even
        for a 1D input (interpreted as a single sample).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before score().")
        feats = _as_numpy(features).astype(np.float64)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: detector expects {self.feature_dim}, "
                f"got {feats.shape[1]}."
            )
        centered = feats - self.mean
        # Solve L y = (x - μ)ᵀ → mahalanobis² = ||y||². np.linalg.solve
        # doesn't know `cov_chol` is triangular, but the matrix is small
        # (d ≤ 2048 in practice) so the wasted O(d³/3) vs triangular
        # solve doesn't matter at inference latency.
        y = np.linalg.solve(self.cov_chol, centered.T)
        mahal_sq = np.sum(y * y, axis=0)
        d = self.feature_dim
        return 0.5 * (mahal_sq + self.log_det + d * np.log(2.0 * np.pi))

    def is_ood(
        self,
        features: torch.Tensor | np.ndarray,
    ) -> list[bool]:
        """Per-sample OOD verdict using the calibrated threshold."""
        if self.threshold is None:
            raise RuntimeError(
                "Call calibrate_threshold() before is_ood()."
            )
        scores = self.score(features)
        return [bool(s > self.threshold) for s in scores]

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        if not self.is_fitted:
            raise RuntimeError("OODDetector is not fitted; nothing to save.")
        return {
            "method": self.method,
            "feature_dim": int(self.feature_dim),
            "mean": self.mean.tolist(),
            "cov_chol": self.cov_chol.tolist(),
            "log_det": float(self.log_det),
            "regularization": float(self.regularization),
            "threshold": None if self.threshold is None else float(self.threshold),
            "percentile": float(self.percentile),
            "n_train": int(self.n_train),
        }

    @classmethod
    def from_dict(cls, d: dict) -> OODDetector:
        return cls(
            method=str(d.get("method", "gaussian")),
            feature_dim=int(d.get("feature_dim", 0)),
            mean=np.asarray(d["mean"], dtype=np.float64) if "mean" in d else None,
            cov_chol=(
                np.asarray(d["cov_chol"], dtype=np.float64)
                if "cov_chol" in d
                else None
            ),
            log_det=None if d.get("log_det") is None else float(d["log_det"]),
            regularization=float(d.get("regularization", 1e-3)),
            threshold=None if d.get("threshold") is None else float(d["threshold"]),
            percentile=float(d.get("percentile", 95.0)),
            n_train=int(d.get("n_train", 0)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> OODDetector:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
