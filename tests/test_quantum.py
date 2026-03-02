"""Tests for the quantum circuit components."""

import numpy as np
from pennylane import numpy as pnp

from medqcnn.quantum.ansatz import get_param_shape
from medqcnn.quantum.qnode import create_qnode


class TestQNode:
    def test_qnode_output_range(self):
        """QNode output (averaged ⟨σ_z⟩) must be in [-1, 1]."""
        qnode = create_qnode(n_qubits=4, n_layers=2)

        # Random L2-normalized features
        features = np.random.rand(2**4)
        features = features / np.linalg.norm(features)

        params = pnp.random.randn(2, 4, 2) * 0.1
        result = qnode(features, params)

        assert -1.0 <= float(result) <= 1.0

    def test_param_shape(self):
        shape = get_param_shape(n_layers=4, n_qubits=8)
        assert shape == (4, 8, 2)


class TestObservables:
    def test_averaged_hamiltonian(self):
        from medqcnn.quantum.observables import averaged_pauli_z

        hamiltonian = averaged_pauli_z(wires=range(4))
        # Should have 4 terms with equal coefficients
        coeffs = hamiltonian.coeffs
        assert len(coeffs) == 4
        assert all(abs(c - 0.25) < 1e-10 for c in coeffs)
