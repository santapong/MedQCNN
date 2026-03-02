"""Tests for the classical CNN backbone and projector."""

import torch

from medqcnn.config.constants import LATENT_DIM


class TestClassicalBackbone:
    def test_output_shape(self):
        from medqcnn.classical.backbone import ClassicalBackbone

        model = ClassicalBackbone(pretrained=False, freeze=True)
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, model.feature_dim)

    def test_frozen_params(self):
        from medqcnn.classical.backbone import ClassicalBackbone

        model = ClassicalBackbone(pretrained=False, freeze=True)
        for param in model.features.parameters():
            assert not param.requires_grad


class TestLatentProjector:
    def test_output_shape(self):
        from medqcnn.classical.projector import LatentProjector

        projector = LatentProjector(input_dim=512)
        x = torch.randn(4, 512)
        z = projector(x)
        assert z.shape == (4, LATENT_DIM)

    def test_l2_normalized(self):
        from medqcnn.classical.projector import LatentProjector

        projector = LatentProjector(input_dim=512)
        x = torch.randn(4, 512)
        z = projector(x)
        norms = torch.norm(z, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)
