"""Tests for the hybrid model assembly."""


class TestHybridQCNN:
    def test_count_trainable_params(self):
        from medqcnn.model.hybrid import HybridQCNN

        model = HybridQCNN(n_qubits=4, n_layers=2, n_classes=2, pretrained=False)
        param_counts = model.count_trainable_params()

        assert "projector" in param_counts
        assert "quantum" in param_counts
        assert "classifier" in param_counts
        assert "total" in param_counts
        assert param_counts["total"] > 0

        # The backbone should NOT appear in trainable params
        backbone_params = sum(
            p.numel() for p in model.backbone.parameters() if p.requires_grad
        )
        assert backbone_params == 0
