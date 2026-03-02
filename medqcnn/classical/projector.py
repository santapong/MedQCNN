"""
Projection head: FC layer mapping backbone features → R^{2^n}.

This is the critical bridge between the classical and quantum
domains. The output vector z must be L2-normalized (||z||₂ = 1)
to serve as valid amplitudes for quantum state preparation.

Sprint 2 upgrade: Added BatchNorm1d before L2 normalization for
training stability, and Kaiming initialization for faster convergence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from medqcnn.config.constants import LATENT_DIM


class LatentProjector(nn.Module):
    """Fully-connected projection to the quantum-compatible latent space.

    Maps the backbone feature vector to R^{latent_dim} and applies
    L2 normalization to satisfy the quantum amplitude constraint.

    Architecture:
        Linear(input_dim, 512) → BatchNorm → ReLU → Dropout →
        Linear(512, latent_dim) → BatchNorm → L2Normalize

    Args:
        input_dim: Dimensionality of the backbone output.
        latent_dim: Target dimensionality (must equal 2^NUM_QUBITS).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = LATENT_DIM,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

        # Kaiming initialization for ReLU layers
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming uniform initialization to Linear layers."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to the latent space and L2-normalize.

        Args:
            x: Feature tensor of shape (B, input_dim).

        Returns:
            L2-normalized tensor of shape (B, latent_dim).
            Each row satisfies ||z||₂ = 1.
        """
        z = self.projection(x)
        z = functional.normalize(z, p=2, dim=-1)  # ||z||₂ = 1
        return z
