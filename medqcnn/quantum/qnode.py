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

Noise harness (Phase 7): each factory accepts an optional
`noise_config: NoiseConfig`. When set, the device switches to
`qiskit.aer` with the corresponding noise model and finite shots,
and the differentiation method falls back to parameter-shift —
the only PennyLane diff method compatible with sampling devices.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from medqcnn.config.constants import NUM_ANSATZ_LAYERS, NUM_QUBITS, NUM_SHOTS_NOISY
from medqcnn.quantum.ansatz import get_param_shape, hardware_efficient_ansatz
from medqcnn.quantum.encoding import amplitude_encode
from medqcnn.quantum.noise import NoiseConfig, build_noise_model


def _build_device(
    n_qubits: int,
    noise_config: NoiseConfig | None,
    device_name: str,
    shots: int | None,
):
    """Construct the PennyLane device, routing through Aer when noisy.

    Returns a tuple ``(device, effective_diff_method_override)`` where
    the override is ``"parameter-shift"`` when the device is noisy
    (sampling), else ``None`` (caller keeps its requested diff method).
    """
    if noise_config is None or noise_config.is_noiseless:
        return qml.device(device_name, wires=n_qubits), None

    noise_model = build_noise_model(noise_config)
    aer_device = qml.device(
        "qiskit.aer",
        wires=n_qubits,
        shots=shots if shots is not None else NUM_SHOTS_NOISY,
        noise_model=noise_model,
    )
    return aer_device, "parameter-shift"


def create_qnode(
    n_qubits: int = NUM_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    diff_method: str = "parameter-shift",
    device_name: str = "default.qubit",
    noise_config: NoiseConfig | None = None,
    shots: int | None = None,
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
            Forced to "parameter-shift" when `noise_config` is set.
        device_name: PennyLane device backend (ignored when noisy).
        noise_config: Optional Aer noise model. When provided and
            non-trivial, the device becomes `qiskit.aer` with finite
            shots and the supplied noise model.
        shots: Measurement shots used by the noisy device. Ignored
            when `noise_config` is None / noiseless.

    Returns:
        A QNode callable: (features, params) → scalar expectation value.
    """
    dev, diff_override = _build_device(n_qubits, noise_config, device_name, shots)
    effective_diff = diff_override or diff_method
    wires = range(n_qubits)

    @qml.qnode(dev, diff_method=effective_diff)
    def circuit(features: pnp.ndarray, params: pnp.ndarray) -> float:
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
    noise_config: NoiseConfig | None = None,
    shots: int | None = None,
) -> tuple[qml.QNode, dict[str, tuple[int, ...]]]:
    """Create a QNode compatible with PennyLane's TorchLayer.

    On `default.qubit` (no noise) the QNode uses backprop. With a
    noise model the QNode switches to the Aer sampling device and
    parameter-shift gradients — backprop is not defined for
    sampling-based devices.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz layers.
        device_name: PennyLane device backend (ignored when noisy).
        noise_config: Optional Aer noise model.
        shots: Measurement shots when noisy.

    Returns:
        Tuple of (qnode, weight_shapes) where weight_shapes maps
        parameter names to their shapes for TorchLayer initialization.
    """
    dev, diff_override = _build_device(n_qubits, noise_config, device_name, shots)
    effective_diff = diff_override or "backprop"
    wires = range(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=effective_diff)
    def circuit(inputs, weights):
        amplitude_encode(inputs, wires=wires)
        hardware_efficient_ansatz(weights, wires=wires, n_layers=n_layers)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    weight_shapes = {"weights": get_param_shape(n_layers, n_qubits)}

    return circuit, weight_shapes


def create_quantum_layer(
    n_qubits: int = NUM_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    device_name: str = "default.qubit",
    noise_config: NoiseConfig | None = None,
    shots: int | None = None,
):
    """Create a PennyLane TorchLayer wrapping the quantum circuit.

    Args:
        n_qubits: Number of qubits.
        n_layers: Number of ansatz layers.
        device_name: PennyLane device backend (ignored when noisy).
        noise_config: Optional Aer noise model. See `create_torch_qnode`.
        shots: Measurement shots when noisy.

    Returns:
        A `qml.qnn.TorchLayer` instance (subclass of nn.Module).
    """
    qnode, weight_shapes = create_torch_qnode(
        n_qubits=n_qubits,
        n_layers=n_layers,
        device_name=device_name,
        noise_config=noise_config,
        shots=shots,
    )

    return qml.qnn.TorchLayer(qnode, weight_shapes)
