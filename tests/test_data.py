"""Tests for the data loading and preprocessing module."""

import numpy as np

from medqcnn.data.preprocessing import (
    extract_axial_slices,
    normalize_intensity,
    preprocess_pipeline,
    resize_slice,
)


class TestNormalizeIntensity:
    def test_output_range(self):
        image = np.random.rand(64, 64).astype(np.float32) * 1000
        result = normalize_intensity(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_constant_image(self):
        image = np.ones((32, 32), dtype=np.float32) * 42.0
        result = normalize_intensity(image)
        assert np.allclose(result, 0.0)


class TestResizeSlice:
    def test_resize_dimensions(self):
        image = np.random.rand(100, 150).astype(np.float32)
        result = resize_slice(image, (224, 224))
        assert result.shape == (224, 224)


class TestExtractAxialSlices:
    def test_middle_extraction(self):
        volume = np.random.rand(64, 64, 20).astype(np.float32)
        slices = extract_axial_slices(volume, num_slices=1, strategy="middle")
        assert len(slices) == 1
        assert slices[0].shape == (64, 64)

    def test_uniform_extraction(self):
        volume = np.random.rand(64, 64, 20).astype(np.float32)
        slices = extract_axial_slices(volume, num_slices=5, strategy="uniform")
        assert len(slices) == 5


class TestPreprocessPipeline:
    def test_output_shape(self):
        image = np.random.rand(100, 150).astype(np.float32)
        result = preprocess_pipeline(image, target_size=(224, 224))
        assert result.shape == (1, 224, 224)
        assert result.dtype == np.float32
