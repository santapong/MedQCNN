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
from medqcnn.quantum.noise import NoiseConfig
from medqcnn.quantum.qnode import (
    VALID_ENCODINGS,
    create_quantum_layer,
    latent_dim_for_encoding,
)


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
        noise_config: Optional Aer noise model. When provided and
            non-trivial, the quantum layer runs on `qiskit.aer` with
            the noise model and parameter-shift gradients (slower but
            NISQ-credible).
        eval_noise_config: Optional separate noise model used only when
            the module is in `eval()` mode. Enables the "noise as
            regulariser" pattern (arXiv:2601.13275): train with
            depolarising noise injected, evaluate on a clean circuit.
            When set, a second TorchLayer is constructed with
            ``requires_grad=False`` parameters and the trainable
            weights are copied across just before each eval forward.
    """

    def __init__(
        self,
        n_qubits: int = NUM_QUBITS,
        n_layers: int = NUM_ANSATZ_LAYERS,
        n_classes: int = 2,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = True,
        noise_config: NoiseConfig | None = None,
        eval_noise_config: NoiseConfig | None = None,
        encoding: str = "amplitude",
    ) -> None:
        super().__init__()

        if encoding not in VALID_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {VALID_ENCODINGS}; got {encoding!r}"
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.noise_config = noise_config
        self.eval_noise_config = eval_noise_config
        latent_dim = latent_dim_for_encoding(encoding, n_qubits, n_layers)

        # --- Classical components (Node A) ---
        self.backbone = ClassicalBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze=True,
        )
        # Amplitude encoding needs ||z||=1; data re-uploading uses raw
        # rotation angles, so the projector skips the L2 step.
        self.projector = LatentProjector(
            input_dim=self.backbone.feature_dim,
            latent_dim=latent_dim,
            normalize=(encoding == "amplitude"),
        )

        # --- Quantum circuit (Node B) ---
        # TorchLayer wraps the QNode as a native nn.Module.
        # Gradients flow through automatically via PyTorch autograd.
        self.quantum_layer = create_quantum_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            noise_config=noise_config,
            encoding=encoding,
        )

        # Optional second layer used only at eval time. Its parameters
        # are frozen; we copy weights from `quantum_layer` before each
        # eval forward so the optimizer never sees a stale copy.
        self._eval_quantum_layer: nn.Module | None = None
        if self._needs_separate_eval_layer():
            self._eval_quantum_layer = create_quantum_layer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                noise_config=eval_noise_config,
                encoding=encoding,
            )
            for p in self._eval_quantum_layer.parameters():
                p.requires_grad = False

        # --- Classification head ---
        # Input: n_qubits expectation values from quantum layer
        # Output: class logits
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, n_classes),
        )

    def _needs_separate_eval_layer(self) -> bool:
        """True when train and eval noise specs differ meaningfully."""
        if self.eval_noise_config is None:
            return False
        train_noiseless = self.noise_config is None or self.noise_config.is_noiseless
        eval_noiseless = self.eval_noise_config.is_noiseless
        if train_noiseless and eval_noiseless:
            return False
        return self.noise_config != self.eval_noise_config

    def _sync_eval_weights(self) -> None:
        """Copy trainable quantum weights into the frozen eval layer."""
        if self._eval_quantum_layer is None:
            return
        with torch.no_grad():
            self._eval_quantum_layer.weights.copy_(self.quantum_layer.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full hybrid pipeline.

        The entire pipeline is differentiable — gradients propagate
        from the loss through the classifier, quantum layer, and
        projector. Only the ResNet backbone is frozen.

        When `eval_noise_config` is set and the module is in
        ``eval()`` mode, the frozen eval-layer (with its own
        noise model) runs instead of the trainable layer. Weights
        are synced from the trainable layer immediately before the
        forward, so eval always reflects the latest training step.

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
        if self._eval_quantum_layer is not None and not self.training:
            self._sync_eval_weights()
            q_out = self._eval_quantum_layer(z)
        else:
            q_out = self.quantum_layer(z)

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
