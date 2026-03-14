"""
Image preprocessing utilities for MedQCNN.

Provides intensity normalization, resizing, slice extraction from 3D volumes,
and a full preprocessing pipeline for medical images.

Note: These utilities are designed for raw image ingestion paths (DICOM, NIfTI,
raw numpy arrays) rather than MedMNIST datasets, which use torchvision transforms
in the data loader. Both paths produce ResNet-compatible 224x224 inputs.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalize image intensity to [0, 1] range.

    Uses min-max normalization. If the image is constant (max == min),
    returns an all-zero array.

    Args:
        image: Input image as a NumPy array (any dtype).

    Returns:
        Normalized float32 array with values in [0, 1].
    """
    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_val) / (max_val - min_val)


def resize_slice(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize a 2D image slice to the target spatial dimensions.

    Args:
        image: 2D input image (H, W).
        target_size: Target (height, width).

    Returns:
        Resized float32 array.
    """
    resized = cv2.resize(
        image.astype(np.float32),
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32)


def extract_axial_slices(
    volume: np.ndarray,
    num_slices: int = 1,
    strategy: str = "middle",
) -> list[np.ndarray]:
    """Extract 2D axial slices from a 3D volume.

    Args:
        volume: 3D array of shape (H, W, D) where D is the depth axis.
        num_slices: Number of slices to extract.
        strategy: Extraction strategy — "middle" or "uniform".

    Returns:
        List of 2D slices as NumPy arrays.
    """
    depth = volume.shape[2]

    if strategy == "middle":
        center = depth // 2
        half = num_slices // 2
        start = max(0, center - half)
        indices = list(range(start, min(start + num_slices, depth)))
    elif strategy == "uniform":
        indices = np.linspace(0, depth - 1, num=num_slices, dtype=int).tolist()
    else:
        msg = f"Unknown strategy: {strategy}. Use 'middle' or 'uniform'."
        raise ValueError(msg)

    return [volume[:, :, idx] for idx in indices]


def preprocess_pipeline(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Full preprocessing pipeline: normalize → resize → add channel dim.

    Args:
        image: 2D input image (H, W).
        target_size: Target spatial dimensions.

    Returns:
        Preprocessed array of shape (1, H, W) as float32.
    """
    normalized = normalize_intensity(image)
    resized = resize_slice(normalized, target_size)
    # Add channel dimension: (H, W) → (1, H, W)
    return np.expand_dims(resized, axis=0)
