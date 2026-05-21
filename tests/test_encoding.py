"""Tests for the data re-uploading encoding scheme and HybridQCNN integration."""

from __future__ import annotations

import pytest
import torch

from medqcnn.model.hybrid import HybridQCNN
from medqcnn.quantum.encoding import data_reupload_latent_dim
from medqcnn.quantum.qnode import (
    VALID_ENCODINGS,
    create_quantum_layer,
    latent_dim_for_encoding,
)


class TestLatentDimResolution:
    def test_amplitude_encoding_is_two_to_the_n(self):
        assert latent_dim_for_encoding("amplitude", 4, 2) == 16
        assert latent_dim_for_encoding("amplitude", 8, 4) == 256

    def test_reupload_encoding_is_layers_times_qubits(self):
        assert latent_dim_for_encoding("reupload", 4, 2) == 8
        assert latent_dim_for_encoding("reupload", 4, 4) == 16
        assert latent_dim_for_encoding("reupload", 8, 4) == 32

    def test_reupload_helper_matches(self):
        assert data_reupload_latent_dim(4, 4) == 16

    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError):
            latent_dim_for_encoding("kitchen-sink", 4, 4)


class TestQuantumLayerFactory:
    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError):
            create_quantum_layer(n_qubits=4, n_layers=2, encoding="garbage")

    def test_reupload_forward_runs(self):
        n_qubits, n_layers = 4, 2
        layer = create_quantum_layer(
            n_qubits=n_qubits, n_layers=n_layers, encoding="reupload"
        )
        latent = torch.randn(3, n_layers * n_qubits)
        out = layer(latent)
        assert out.shape == (3, n_qubits)

    def test_amplitude_forward_runs(self):
        n_qubits, n_layers = 4, 2
        layer = create_quantum_layer(
            n_qubits=n_qubits, n_layers=n_layers, encoding="amplitude"
        )
        z = torch.randn(2, 2**n_qubits)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        out = layer(z)
        assert out.shape == (2, n_qubits)


class TestHybridQCNNWithReupload:
    @pytest.fixture(scope="class")
    def model(self) -> HybridQCNN:
        torch.manual_seed(0)
        return HybridQCNN(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            pretrained=False,
            encoding="reupload",
        )

    def test_projector_dim_matches_reupload_latent(self, model: HybridQCNN):
        # latent_dim should equal n_layers * n_qubits = 2 * 4 = 8
        # The projector's last Linear maps to that dim.
        last_linear = None
        for module in model.projector.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        assert last_linear is not None
        assert last_linear.out_features == 8

    def test_projector_is_unnormalised(self, model: HybridQCNN):
        assert model.projector.normalize is False

    def test_forward_shape_is_n_classes(self, model: HybridQCNN):
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 2)

    def test_amplitude_default_still_works(self):
        m = HybridQCNN(n_qubits=4, n_layers=2, n_classes=2, pretrained=False)
        assert m.encoding == "amplitude"
        x = torch.randn(2, 3, 224, 224)  # bs > 1 so BatchNorm in train mode is happy
        assert m(x).shape == (2, 2)


class TestEncodingChoiceMatrix:
    @pytest.mark.parametrize("encoding", list(VALID_ENCODINGS))
    def test_constructs(self, encoding: str):
        m = HybridQCNN(
            n_qubits=4, n_layers=2, n_classes=2, pretrained=False, encoding=encoding
        )
        assert m.encoding == encoding
