"""
HybridQCNN: End-to-end hybrid quantum-classical model.

Combines the classical backbone (Node A) with the quantum circuit
(Node B) into a single differentiable PyTorch nn.Module:

    Input Image → ResNet backbone → FC projector → L2 norm →
    Amplitude Encoding → Variational Ansatz → ⟨σ_z⟩ → Output

Sprint 2 upgrade: Quantum parameters are now trained via PennyLane's
TorchLayer, enabling native PyTorch autograd gradient flow through
the quantum circuit. No more .detach().numpy() — the entire pipeline
is fully differentiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from medqcnn.classical.backbone import ClassicalBackbone
from medqcnn.classical.projector import LatentProjector
from medqcnn.config.constants import (
    BACKBONE_NAME,
    NUM_ANSATZ_LAYERS,
    NUM_QUBITS,
)
from medqcnn.quantum.qnode import create_quantum_layer


class HybridQCNN(nn.Module):
    """Hybrid Quantum-Classical Convolutional Neural Network.

    This is the top-level model that chains:
      1. Classical feature extraction (frozen ResNet)
      2. Latent projection to R^{2^n} with L2 normalization
      3. Quantum circuit evaluation via TorchLayer (differentiable!)
      4. Post-processing head for classification

    The quantum layer outputs `n_qubits` expectation values (one per
    qubit) instead of a single averaged scalar. This gives the
    classifier richer features to work with.

    Args:
        n_qubits: Number of qubits for the quantum circuit.
        n_layers: Number of variational ansatz layers.
        n_classes: Number of output classes (2 for binary classification).
        backbone_name: Pre-trained backbone architecture name.
        pretrained: Whether to load pre-trained backbone weights.
    """

    def __init__(
        self,
        n_qubits: int = NUM_QUBITS,
        n_layers: int = NUM_ANSATZ_LAYERS,
        n_classes: int = 2,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        latent_dim = 2**n_qubits

        # --- Classical components (Node A) ---
        self.backbone = ClassicalBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=True,
        )
        self.projector = LatentProjector(
            input_dim=self.backbone.feature_dim,
            latent_dim=latent_dim,
        )

        # --- Quantum circuit (Node B) ---
        # TorchLayer wraps the QNode as a native nn.Module.
        # Gradients flow through automatically via PyTorch autograd.
        self.quantum_layer = create_quantum_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
        )

        # --- Classification head ---
        # Input: n_qubits expectation values from quantum layer
        # Output: class logits
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full hybrid pipeline.

        The entire pipeline is differentiable — gradients propagate
        from the loss through the classifier, quantum layer, and
        projector. Only the ResNet backbone is frozen.

        Args:
            x: Input image batch of shape (B, C, H, W).

        Returns:
            Class logits of shape (B, n_classes).
        """
        # Step 1: Classical feature extraction (frozen)
        features = self.backbone(x)  # (B, feature_dim)

        # Step 2: Project to quantum-compatible latent space
        z = self.projector(features)  # (B, 2^n_qubits), L2-normalized

        # Step 3: Quantum circuit evaluation via TorchLayer
        # TorchLayer processes each sample in the batch automatically.
        # Input: (B, 2^n_qubits) → Output: (B, n_qubits)
        q_out = self.quantum_layer(z)  # (B, n_qubits)

        # Step 4: Classification head
        logits = self.classifier(q_out)  # (B, n_classes)
        return logits

    def count_trainable_params(self) -> dict[str, int]:
        """Count trainable parameters by component.

        Returns:
            Dict mapping component name → param count.
        """
        projector_params = sum(
            p.numel() for p in self.projector.parameters() if p.requires_grad
        )
        quantum_params = sum(
            p.numel() for p in self.quantum_layer.parameters() if p.requires_grad
        )
        classifier_params = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )

        return {
            "projector": projector_params,
            "quantum": quantum_params,
            "classifier": classifier_params,
            "total": projector_params + quantum_params + classifier_params,
        }
