"""
HybridQCNN: End-to-end hybrid quantum-classical model.

Combines the classical backbone (Node A) with the quantum circuit
(Node B) into a single differentiable PyTorch nn.Module:

    Input Image → ResNet backbone → FC projector → L2 norm →
    Amplitude Encoding → Variational Ansatz → ⟨σ_z⟩ → Output

The quantum parameters are optimized jointly with the projection
head via the parameter-shift rule. The ResNet backbone remains
frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medqcnn.classical.backbone import ClassicalBackbone
from medqcnn.classical.projector import LatentProjector
from medqcnn.config.constants import (
    BACKBONE_NAME,
    LATENT_DIM,
    NUM_ANSATZ_LAYERS,
    NUM_QUBITS,
)
from medqcnn.quantum.ansatz import get_param_shape
from medqcnn.quantum.qnode import create_qnode


class HybridQCNN(nn.Module):
    """Hybrid Quantum-Classical Convolutional Neural Network.

    This is the top-level model that chains:
      1. Classical feature extraction (frozen ResNet)
      2. Latent projection to R^256 with L2 normalization
      3. Quantum circuit evaluation (amplitude encoding → HEA → ⟨σ_z⟩)
      4. Post-processing head for classification

    Args:
        n_qubits: Number of qubits for the quantum circuit.
        n_layers: Number of variational ansatz layers.
        n_classes: Number of output classes (2 for binary classification).
        backbone_name: Pre-trained backbone architecture name.
    """

    def __init__(
        self,
        n_qubits: int = NUM_QUBITS,
        n_layers: int = NUM_ANSATZ_LAYERS,
        n_classes: int = 2,
        backbone_name: str = BACKBONE_NAME,
    ) -> None:
        super().__init__()

        # --- Classical components (Node A) ---
        self.backbone = ClassicalBackbone(
            backbone_name=backbone_name,
            pretrained=True,
            freeze=True,
        )
        self.projector = LatentProjector(
            input_dim=self.backbone.feature_dim,
            latent_dim=LATENT_DIM,
        )

        # --- Quantum components (Node B) ---
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qnode = create_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        # Trainable quantum parameters
        param_shape = get_param_shape(n_layers, n_qubits)
        self.quantum_params = nn.Parameter(
            torch.randn(param_shape) * 0.1  # small random init
        )

        # --- Post-processing head ---
        # Maps quantum expectation value(s) → class logits
        self.classifier = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full hybrid pipeline.

        Args:
            x: Input image batch of shape (B, C, H, W).

        Returns:
            Class logits of shape (B, n_classes).
        """
        batch_size = x.shape[0]

        # Step 1: Classical feature extraction
        features = self.backbone(x)  # (B, feature_dim)

        # Step 2: Project to quantum-compatible latent space
        z = self.projector(features)  # (B, 256), L2-normalized

        # Step 3: Quantum circuit evaluation (per-sample)
        q_outputs = []
        for i in range(batch_size):
            z_i = z[i].detach().cpu().numpy()
            params_np = self.quantum_params.detach().cpu().numpy()
            expval = self.qnode(z_i, params_np)
            q_outputs.append(float(expval))

        q_tensor = torch.tensor(
            q_outputs, dtype=torch.float32, device=x.device
        ).unsqueeze(1)  # (B, 1)

        # Step 4: Classification head
        logits = self.classifier(q_tensor)  # (B, n_classes)
        return logits

    def count_trainable_params(self) -> dict[str, int]:
        """Count trainable parameters by component.

        Returns:
            Dict mapping component name → param count.
        """
        projector_params = sum(
            p.numel() for p in self.projector.parameters() if p.requires_grad
        )
        quantum_params = self.quantum_params.numel()
        classifier_params = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )

        return {
            "projector": projector_params,
            "quantum": quantum_params,
            "classifier": classifier_params,
            "total": projector_params + quantum_params + classifier_params,
        }
