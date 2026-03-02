"""
Projection head: FC layer mapping backbone features → R^256.

This is the critical bridge between the classical and quantum
domains. The output vector z must be L2-normalized (||z||₂ = 1)
to serve as valid amplitudes for quantum state preparation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from medqcnn.config.constants import LATENT_DIM


class LatentProjector(nn.Module):
    """Fully-connected projection to the quantum-compatible latent space.

    Maps the backbone feature vector to R^{LATENT_DIM} and applies
    L2 normalization to satisfy the quantum amplitude constraint.

    Args:
        input_dim: Dimensionality of the backbone output.
        latent_dim: Target dimensionality (must equal 2^NUM_QUBITS).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, latent_dim),
        )

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
