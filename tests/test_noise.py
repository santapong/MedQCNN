"""Tests for the Aer noise harness."""

from __future__ import annotations

import numpy as np
import pytest
from pennylane import numpy as pnp

from medqcnn.quantum import NOISE_PRESETS, NoiseConfig, build_noise_model, get_preset
from medqcnn.quantum.qnode import create_qnode


class TestNoiseConfig:
    def test_default_is_noiseless(self):
        assert NoiseConfig().is_noiseless

    def test_low_preset_is_not_noiseless(self):
        assert not NOISE_PRESETS["low"].is_noiseless

    def test_to_dict_round_trip(self):
        cfg = NoiseConfig(depolarising_p=0.005, readout_p=0.02)
        assert cfg.to_dict() == {"depolarising_p": 0.005, "readout_p": 0.02}

    @pytest.mark.parametrize("p", [-0.01, 1.5])
    def test_rejects_out_of_range(self, p):
        with pytest.raises(ValueError):
            NoiseConfig(depolarising_p=p, readout_p=0.0)

    def test_get_preset_unknown_lists_options(self):
        with pytest.raises(KeyError, match="Valid:"):
            get_preset("not-a-preset")


class TestBuildNoiseModel:
    def test_noiseless_returns_none(self):
        assert build_noise_model(NoiseConfig()) is None

    def test_noisy_returns_qiskit_aer_model(self):
        pytest.importorskip("qiskit_aer")
        from qiskit_aer.noise import NoiseModel

        nm = build_noise_model(NOISE_PRESETS["medium"])
        assert isinstance(nm, NoiseModel)
        instructions = set(nm.noise_instructions)
        assert "ry" in instructions
        assert "cz" in instructions
        assert "measure" in instructions


class TestNoisyQNode:
    """End-to-end: a noisy QNode runs forward on Aer."""

    def test_qnode_runs_with_noise(self):
        pytest.importorskip("qiskit_aer")

        n_qubits = 4
        n_layers = 2
        qnode = create_qnode(
            n_qubits=n_qubits,
            n_layers=n_layers,
            noise_config=NOISE_PRESETS["low"],
            shots=64,
        )

        features = np.random.RandomState(0).rand(2**n_qubits)
        features = features / np.linalg.norm(features)
        params = pnp.zeros((n_layers, n_qubits, 2))

        result = float(qnode(features, params))
        assert -1.0 <= result <= 1.0


class TestNoiseAsRegulariser:
    """Idea 2: HybridQCNN can train noisy and eval clean."""

    def test_eval_layer_only_built_when_configs_differ(self):
        from medqcnn.model.hybrid import HybridQCNN

        # Same train + eval noise (none): no separate eval layer.
        m_same = HybridQCNN(n_qubits=4, n_layers=2, n_classes=2, pretrained=False)
        assert m_same._eval_quantum_layer is None

        # Different specs (clean eval, no Aer needed): separate eval layer.
        m_diff = HybridQCNN(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            pretrained=False,
            noise_config=None,
            eval_noise_config=NoiseConfig(0.0, 0.0),
        )
        # Both noiseless → still considered the same effective config.
        assert m_diff._eval_quantum_layer is None

    def test_eval_layer_params_frozen(self):
        pytest.importorskip("qiskit_aer")
        from medqcnn.model.hybrid import HybridQCNN

        model = HybridQCNN(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            pretrained=False,
            noise_config=NOISE_PRESETS["low"],
            eval_noise_config=NoiseConfig(0.0, 0.0),
        )
        assert model._eval_quantum_layer is not None
        for p in model._eval_quantum_layer.parameters():
            assert not p.requires_grad

    def test_sync_eval_weights_copies_from_train_layer(self):
        pytest.importorskip("qiskit_aer")
        import torch

        from medqcnn.model.hybrid import HybridQCNN

        model = HybridQCNN(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            pretrained=False,
            noise_config=NOISE_PRESETS["low"],
            eval_noise_config=NoiseConfig(0.0, 0.0),
        )
        # Mutate the trainable weights, then sync.
        with torch.no_grad():
            model.quantum_layer.weights.fill_(0.5)
        model._sync_eval_weights()
        assert torch.allclose(
            model._eval_quantum_layer.weights,
            model.quantum_layer.weights,
        )


class TestGradientVariance:
    """Per-ansatz-layer gradient variance helper."""

    def test_returns_nan_when_no_grad(self):
        from medqcnn.quantum.qnode import create_quantum_layer
        from medqcnn.training.metrics import compute_quantum_gradient_variance

        layer = create_quantum_layer(n_qubits=4, n_layers=2)
        variances = compute_quantum_gradient_variance(layer, n_layers=2, n_qubits=4)
        assert variances == [float("nan"), float("nan")] or all(
            v != v for v in variances
        )

    def test_returns_one_value_per_layer(self):
        import torch

        from medqcnn.quantum.qnode import create_quantum_layer
        from medqcnn.training.metrics import compute_quantum_gradient_variance

        n_qubits, n_layers = 4, 3
        layer = create_quantum_layer(n_qubits=n_qubits, n_layers=n_layers)

        z = torch.rand(2, 2**n_qubits)
        z = z / z.norm(dim=1, keepdim=True)
        out = layer(z).sum()
        out.backward()

        variances = compute_quantum_gradient_variance(
            layer, n_layers=n_layers, n_qubits=n_qubits
        )
        assert len(variances) == n_layers
        # At least one layer should have a real (non-NaN) variance.
        assert any(v == v for v in variances)
