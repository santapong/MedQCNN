"""
Classical-to-quantum data encoding strategies.

Two schemes are supported:

* :func:`amplitude_encode` — the canonical map ``|ψ(z)⟩ = Σ z_i |i⟩``
  over an L2-normalised latent of length ``2^n``. One-shot, exponential
  compression, no extra params.
* :func:`data_reupload_encode` — interleaves a per-qubit ``RY(z)``
  rotation with the variational ansatz at every layer (Pérez-Salinas
  et al., 2020; reinforced by the May 2025 medRxiv data-re-uploading
  benchmark on lab-medicine datasets). Trades the exponential
  compression for higher expressivity per qubit; latent size shrinks
  from ``2^n`` to ``n_layers * n_qubits``.

The data-reupload variant is the lever that addresses ``research.md``
§1 (expressivity vs ansatz depth) — fixed param count, more bandwidth
on each qubit.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp


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


def data_reupload_encode(
    features: pnp.ndarray,
    params: pnp.ndarray,
    wires: range | list[int],
    n_layers: int,
) -> None:
    """Apply a data re-uploading ansatz interleaving data + trainables.

    Circuit (per layer ``l``):

        for q in wires:
            RY(features[l, q])           # data injection (no params)
        for q in wires:
            RY(params[l, q, 0])          # trainable
            RZ(params[l, q, 1])
        ring of CZ entanglers

    The data tensor has shape ``(n_layers, n_qubits)`` and is **not**
    L2-normalised — these are rotation angles, not amplitudes. The
    classical projector is responsible for producing values in a sane
    range (a Tanh or a small linear layer is enough).

    Args:
        features: Tensor of shape ``(n_layers, n_qubits)`` carrying the
            data values for each layer's rotation.
        params: Trainable tensor of shape ``(n_layers, n_qubits, 2)``.
        wires: Qubit indices.
        n_layers: Number of (data + ansatz) layers.
    """
    n_qubits = len(wires)
    for layer in range(n_layers):
        # Data injection
        for q in range(n_qubits):
            qml.RY(features[layer, q], wires=wires[q])
        # Trainable HEA layer
        for q in range(n_qubits):
            qml.RY(params[layer, q, 0], wires=wires[q])
            qml.RZ(params[layer, q, 1], wires=wires[q])
        for q in range(n_qubits - 1):
            qml.CZ(wires=[wires[q], wires[q + 1]])
        qml.CZ(wires=[wires[n_qubits - 1], wires[0]])


def data_reupload_latent_dim(n_layers: int, n_qubits: int) -> int:
    """Latent-space dimensionality expected by :func:`data_reupload_encode`."""
    return n_layers * n_qubits
