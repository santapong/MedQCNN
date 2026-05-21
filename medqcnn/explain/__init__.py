"""Explainability tools for HybridQCNN.

* :class:`~medqcnn.explain.gradcam.GradCAM` — classical activation
  attribution on the frozen ResNet backbone, adapted to flow gradient
  through the quantum layer.
* :func:`~medqcnn.explain.quantum_attribution.quantum_attribution` —
  per-qubit sensitivity ``∂⟨σ_z^(i)⟩/∂z_j`` exposing which latent
  dimensions each qubit reads from.
"""

from medqcnn.explain.gradcam import GradCAM
from medqcnn.explain.quantum_attribution import quantum_attribution

__all__ = ["GradCAM", "quantum_attribution"]
