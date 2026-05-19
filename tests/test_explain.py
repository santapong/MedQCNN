"""Tests for explainability tools (Grad-CAM + quantum attribution)."""

from __future__ import annotations

import pytest
import torch

from medqcnn.explain import GradCAM, quantum_attribution
from medqcnn.model.hybrid import HybridQCNN


@pytest.fixture(scope="module")
def tiny_model() -> HybridQCNN:
    """4-qubit, 1-layer model with untrained backbone for fast tests."""
    torch.manual_seed(0)
    return HybridQCNN(n_qubits=4, n_layers=1, n_classes=2, pretrained=False)


class TestGradCAM:
    def test_returns_heatmap_at_input_resolution(self, tiny_model: HybridQCNN):
        x = torch.randn(1, 3, 224, 224)
        with GradCAM(tiny_model) as cam:
            heatmap = cam(x, target_class=0)
        assert heatmap.shape == (224, 224)
        assert heatmap.min().item() >= 0.0
        assert heatmap.max().item() <= 1.0 + 1e-5

    def test_uses_argmax_when_target_class_none(self, tiny_model: HybridQCNN):
        x = torch.randn(1, 3, 224, 224)
        with GradCAM(tiny_model) as cam:
            heatmap = cam(x)
        # Just verifying no exception + valid shape
        assert heatmap.shape == (224, 224)

    def test_rejects_multi_sample_batch(self, tiny_model: HybridQCNN):
        x = torch.randn(2, 3, 224, 224)
        with GradCAM(tiny_model) as cam:
            with pytest.raises(ValueError):
                cam(x)

    def test_hooks_removed_after_close(self, tiny_model: HybridQCNN):
        cam = GradCAM(tiny_model)
        target = cam.target_layer
        before = len(target._forward_hooks)
        cam.close()
        after = len(target._forward_hooks)
        assert after < before


class TestQuantumAttribution:
    def test_jacobian_shape_matches_qubits_x_latent(self, tiny_model: HybridQCNN):
        latent_dim = 2 ** tiny_model.n_qubits
        z = torch.randn(latent_dim)
        z = z / z.norm()
        jac = quantum_attribution(tiny_model, z)
        assert jac.shape == (tiny_model.n_qubits, latent_dim)

    def test_accepts_batched_input_of_size_one(self, tiny_model: HybridQCNN):
        latent_dim = 2 ** tiny_model.n_qubits
        z = torch.randn(1, latent_dim)
        z = z / z.norm(dim=-1, keepdim=True)
        jac = quantum_attribution(tiny_model, z)
        assert jac.shape == (tiny_model.n_qubits, latent_dim)

    def test_rejects_non_quantum_models(self):
        class Plain(torch.nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(AttributeError):
            quantum_attribution(Plain(), torch.zeros(4))
