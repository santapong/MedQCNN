"""
Variational Ansatz: Hardware-Efficient Ansatz (HEA).

Implements the parameterized quantum circuit that acts as the
"Quantum Attention" mechanism from GEMINI.md Phase 2. Uses:
  - Ry(θ) and Rz(θ) single-qubit rotations
  - Nearest-neighbor CZ entangling gates

The ansatz is kept deliberately shallow (NUM_ANSATZ_LAYERS ≤ 6)
to mitigate barren plateaus: Var(∂L/∂θ) ∝ 2^{-n}.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from medqcnn.config.constants import NUM_ANSATZ_LAYERS, NUM_QUBITS


def hardware_efficient_ansatz(
    params: pnp.ndarray,
    wires: range | list[int],
    n_layers: int = NUM_ANSATZ_LAYERS,
) -> None:
    """Apply hardware-efficient variational ansatz.

    Each layer consists of:
      1. Ry(θ) rotation on every qubit
      2. Rz(θ) rotation on every qubit
      3. Nearest-neighbor CZ entangling gates (ring topology)

    Args:
        params: Trainable parameters of shape (n_layers, n_qubits, 2).
            params[l, q, 0] = Ry angle, params[l, q, 1] = Rz angle.
        wires: Qubit indices.
        n_layers: Number of variational layers.
    """
    n_qubits = len(wires)

    for layer in range(n_layers):
        # Single-qubit rotations
        for qubit in range(n_qubits):
            qml.RY(params[layer, qubit, 0], wires=wires[qubit])
            qml.RZ(params[layer, qubit, 1], wires=wires[qubit])

        # Nearest-neighbor CZ entanglement (ring topology)
        for qubit in range(n_qubits - 1):
            qml.CZ(wires=[wires[qubit], wires[qubit + 1]])
        # Close the ring
        qml.CZ(wires=[wires[n_qubits - 1], wires[0]])


def get_param_shape(
    n_layers: int = NUM_ANSATZ_LAYERS,
    n_qubits: int = NUM_QUBITS,
) -> tuple[int, int, int]:
    """Return the expected parameter shape for the ansatz.

    Returns:
        Tuple (n_layers, n_qubits, 2) — 2 params per qubit per layer
        (one for Ry, one for Rz).
    """
    return (n_layers, n_qubits, 2)
