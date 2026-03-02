"""
Observable measurements for quantum inference.

Implements local Pauli-Z measurements as specified in GEMINI.md
Phase 3. We explicitly avoid global cost functions to prevent
barren plateaus:

    L(θ) = (1/n) Σ_{i=1}^{n} ⟨ψ| U†(θ) σ_z^{(i)} U(θ) |ψ⟩

Using local observables ensures the gradient variance does not
vanish exponentially with qubit count.
"""

from __future__ import annotations

import pennylane as qml

from medqcnn.config.constants import NUM_QUBITS


def local_pauli_z_observables(
    wires: range | list[int],
) -> list[qml.operation.Observable]:
    """Return a list of local Pauli-Z observables, one per qubit.

    These are used as the measurement basis for the quantum circuit.
    The expectation values are averaged to produce the final output.

    Args:
        wires: Qubit indices.

    Returns:
        List of PauliZ observables.
    """
    return [qml.PauliZ(w) for w in wires]


def averaged_pauli_z(
    wires: range | list[int] | None = None,
) -> qml.Hamiltonian:
    """Construct the averaged local Pauli-Z Hamiltonian.

    H = (1/n) Σ σ_z^{(i)}

    This Hamiltonian serves as the cost function observable,
    producing a scalar expectation value in [-1, 1].

    Args:
        wires: Qubit indices. Defaults to range(NUM_QUBITS).

    Returns:
        PennyLane Hamiltonian for the averaged Pauli-Z.
    """
    if wires is None:
        wires = range(NUM_QUBITS)

    n = len(list(wires))
    coeffs = [1.0 / n] * n
    obs = [qml.PauliZ(w) for w in wires]

    return qml.Hamiltonian(coeffs, obs)
