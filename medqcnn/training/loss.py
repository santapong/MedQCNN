"""
Loss functions for hybrid quantum-classical training.

Provides loss functions compatible with the local-observable
measurement strategy defined in GEMINI.md Phase 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """Combined loss for the hybrid model.

    Uses CrossEntropy for classification with optional L2
    regularization on quantum parameters to encourage
    smooth optimization landscapes.

    Args:
        l2_lambda: L2 regularization coefficient for quantum params.
    """

    def __init__(self, l2_lambda: float = 1e-4) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l2_lambda = l2_lambda

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        quantum_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the hybrid loss.

        Args:
            logits: Model output logits (B, n_classes).
            targets: Ground truth labels (B,).
            quantum_params: Optional quantum parameters for L2 reg.

        Returns:
            Scalar loss tensor.
        """
        loss = self.ce_loss(logits, targets)

        if quantum_params is not None and self.l2_lambda > 0:
            l2_reg = self.l2_lambda * torch.sum(quantum_params**2)
            loss = loss + l2_reg

        return loss
