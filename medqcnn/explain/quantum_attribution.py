"""
Per-qubit sensitivity of the variational quantum layer.

For a quantum head producing per-qubit expectations
``q_i(z) = ⟨ψ(z, θ) | σ_z^(i) | ψ(z, θ)⟩``, this module computes the
Jacobian ``J[i, j] = ∂q_i / ∂z_j`` at a given latent vector ``z``.

This complements Grad-CAM:

* Grad-CAM tells you *which input pixels* drove a prediction.
* Quantum attribution tells you *which latent dimensions* drove each
  qubit, exposing whether the VQC is using its full encoding bandwidth
  or collapsing onto a low-rank subspace — a useful diagnostic when
  comparing amplitude encoding to data re-uploading.
"""

from __future__ import annotations

import torch


def quantum_attribution(
    model: torch.nn.Module,
    z: torch.Tensor,
) -> torch.Tensor:
    """Compute ``∂q_i / ∂z_j`` for one latent vector.

    Args:
        model: Any module exposing ``model.quantum_layer`` whose forward
            consumes an L2-normalised latent and returns per-qubit
            expectations of shape ``(n_qubits,)``.
        z: Latent vector of shape ``(latent_dim,)`` or ``(1, latent_dim)``.
            Will be detached and assumed already L2-normalised.

    Returns:
        Jacobian tensor of shape ``(n_qubits, latent_dim)``.
    """
    if not hasattr(model, "quantum_layer"):
        raise AttributeError(
            "model has no `quantum_layer`; quantum_attribution only "
            "applies to HybridQCNN-style models."
        )

    if z.ndim == 2 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 1:
        raise ValueError(f"z must be 1D or (1, D); got shape {tuple(z.shape)}")

    z = z.detach().clone()
    quantum_layer = model.quantum_layer

    def q(latent: torch.Tensor) -> torch.Tensor:
        # quantum_layer expects a batched input
        return quantum_layer(latent[None, :])[0]

    return torch.autograd.functional.jacobian(q, z, create_graph=False, vectorize=False)
