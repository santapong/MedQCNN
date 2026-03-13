"""Tests for model registry module."""

from __future__ import annotations

import pytest

from medqcnn.model.registry import ModelRegistry


class TestModelRegistry:
    """Tests for model version registry."""

    def test_empty_checkpoint_dir(self, tmp_path):
        registry = ModelRegistry(checkpoint_dir=str(tmp_path))
        assert registry.list_versions() == []
        assert registry.active_version is None

    def test_nonexistent_dir(self, tmp_path):
        registry = ModelRegistry(checkpoint_dir=str(tmp_path / "nonexistent"))
        assert registry.list_versions() == []

    def test_set_active_version_invalid(self, tmp_path):
        registry = ModelRegistry(checkpoint_dir=str(tmp_path))
        with pytest.raises(ValueError, match="not found"):
            registry.set_active_version("v999")

    @pytest.mark.skipif(
        not __import__("pathlib")
        .Path(
            __import__("os").path.expanduser(
                "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            )
        )
        .exists(),
        reason="ResNet weights not cached locally",
    )
    def test_get_model_no_checkpoints(self, tmp_path):
        """Should return a fresh model when no checkpoints exist."""
        registry = ModelRegistry(checkpoint_dir=str(tmp_path))
        model = registry.get_model()
        assert model is not None
