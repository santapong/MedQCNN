"""
PennyLane QNode: the complete quantum circuit definition.

Assembles the full quantum computation pipeline:
  1. Amplitude encoding of classical latent vector
  2. Variational ansatz (HEA)
  3. Local Pauli-Z measurement

This QNode is designed to be called from the HybridQCNN PyTorch
model and supports automatic differentiation via the parameter-shift
rule for quantum gradient backpropagation.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from medqcnn.config.constants import NUM_ANSATZ_LAYERS, NUM_QUBITS
from medqcnn.quantum.ansatz import hardware_efficient_ansatz
from medqcnn.quantum.encoding import amplitude_encode


def create_qnode(
    n_qubits: int = NUM_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    diff_method: str = "parameter-shift",
    device_name: str = "default.qubit",
) -> qml.QNode:
    """Create and return a PennyLane QNode for the hybrid model.

    The circuit performs:
      1. Amplitude encoding: |ψ(z)⟩ = Σ z_i |i⟩
      2. Variational ansatz: U(θ)|ψ(z)⟩
      3. Measurement: (1/n) Σ ⟨σ_z^{(i)}⟩

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz layers.
        diff_method: Differentiation method for gradient computation.
            "parameter-shift" is hardware-compatible (NISQ-ready).
        device_name: PennyLane device backend.

    Returns:
        A QNode callable: (features, params) → scalar expectation value.
    """
    dev = qml.device(device_name, wires=n_qubits)
    wires = range(n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(features: pnp.ndarray, params: pnp.ndarray) -> float:
        """Quantum circuit: encode → ansatz → measure.

        Args:
            features: L2-normalized latent vector, shape (2^n_qubits,).
            params: Variational parameters, shape (n_layers, n_qubits, 2).

        Returns:
            Scalar expectation value in [-1, 1] representing the
            averaged local Pauli-Z measurement. Values closer to -1
            indicate higher probability of malignancy.
        """
        # Step 1: Encode classical data into quantum state
        amplitude_encode(features, wires=wires)

        # Step 2: Apply variational ansatz
        hardware_efficient_ansatz(params, wires=wires, n_layers=n_layers)

        # Step 3: Measure averaged local Pauli-Z
        return qml.expval(
            qml.Hamiltonian(
                [1.0 / n_qubits] * n_qubits,
                [qml.PauliZ(w) for w in wires],
            )
        )

    return circuit
