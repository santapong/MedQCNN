"""
PennyLane QNode: the complete quantum circuit definition.

Assembles the full quantum computation pipeline:
  1. Amplitude encoding of classical latent vector
  2. Variational ansatz (HEA)
  3. Local Pauli-Z measurement

Sprint 2 upgrade: Now supports `qml.qnn.TorchLayer` integration
for native PyTorch autograd through quantum parameters. This
replaces the manual .detach().numpy() approach that broke gradient
backpropagation.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from medqcnn.config.constants import NUM_ANSATZ_LAYERS, NUM_QUBITS
from medqcnn.quantum.ansatz import get_param_shape, hardware_efficient_ansatz
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
            averaged local Pauli-Z measurement.
        """
        amplitude_encode(features, wires=wires)
        hardware_efficient_ansatz(params, wires=wires, n_layers=n_layers)

        return qml.expval(
            qml.Hamiltonian(
                [1.0 / n_qubits] * n_qubits,
                [qml.PauliZ(w) for w in wires],
            )
        )

    return circuit


def create_torch_qnode(
    n_qubits: int = NUM_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    device_name: str = "default.qubit",
) -> tuple[qml.QNode, dict[str, tuple[int, ...]]]:
    """Create a QNode compatible with PennyLane's TorchLayer.

    This is the Sprint 2 upgrade that enables native PyTorch autograd
    through the quantum circuit. The circuit uses `interface='torch'`
    and `diff_method='backprop'` for efficient gradient computation
    on simulators.

    The circuit signature is designed for TorchLayer:
      - `inputs`: the classical features (data-dependent, not trained)
      - `weights`: the variational parameters (trained via autograd)

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz layers.
        device_name: PennyLane device backend.

    Returns:
        Tuple of (qnode, weight_shapes) where weight_shapes maps
        parameter names to their shapes for TorchLayer initialization.
    """
    dev = qml.device(device_name, wires=n_qubits)
    wires = range(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        """TorchLayer-compatible quantum circuit.

        Args:
            inputs: L2-normalized latent vector, shape (2^n_qubits,).
                This is the data input — not a trainable parameter.
            weights: Variational parameters, shape (n_layers, n_qubits, 2).
                These are the trainable quantum weights.

        Returns:
            List of expectation values, one per qubit.
        """
        # Step 1: Encode classical data into quantum state
        amplitude_encode(inputs, wires=wires)

        # Step 2: Apply variational ansatz
        hardware_efficient_ansatz(weights, wires=wires, n_layers=n_layers)

        # Step 3: Return per-qubit Pauli-Z expectations
        # Returning n_qubits values gives the classifier richer features
        # than a single averaged scalar.
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    weight_shapes = {"weights": get_param_shape(n_layers, n_qubits)}

    return circuit, weight_shapes


def create_quantum_layer(
    n_qubits: int = NUM_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    device_name: str = "default.qubit",
):
    """Create a PennyLane TorchLayer wrapping the quantum circuit.

    This is the simplest API — returns a PyTorch-native nn.Module
    that can be dropped into any Sequential or forward() method.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz layers.
        device_name: PennyLane device backend.

    Returns:
        A `qml.qnn.TorchLayer` instance (subclass of nn.Module).
    """
    qnode, weight_shapes = create_torch_qnode(
        n_qubits=n_qubits,
        n_layers=n_layers,
        device_name=device_name,
    )

    return qml.qnn.TorchLayer(qnode, weight_shapes)
