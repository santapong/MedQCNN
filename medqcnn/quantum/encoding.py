"""
Amplitude encoding: classical vector → quantum state.

Maps the L2-normalized latent vector z ∈ R^256 into an 8-qubit
quantum state via amplitude encoding:

    |ψ(z)⟩ = Σ_{i=0}^{255} z_i |i⟩

This is the mathematical bridge from the classical Euclidean space
to the exponentially large Hilbert space H_{2^n}.
"""

from __future__ import annotations

import pennylane as qml


def amplitude_encode(features: list[float], wires: range | list[int]) -> None:
    """Apply amplitude encoding to embed classical data into quantum state.

    Prepares the quantum state |ψ(z)⟩ = Σ z_i |i⟩ where z is the
    L2-normalized latent vector from the classical projector.

    Args:
        features: L2-normalized feature vector of length 2^NUM_QUBITS.
            Must satisfy ||features||₂ = 1.
        wires: Qubit indices to encode onto.

    Note:
        PennyLane's AmplitudeEmbedding handles the state preparation
        decomposition automatically, including normalization verification.
    """
    qml.AmplitudeEmbedding(
        features=features,
        wires=wires,
        normalize=True,  # safety net — re-normalizes if slightly off
        pad_with=0.0,
    )
