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
