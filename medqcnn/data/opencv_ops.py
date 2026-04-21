"""
OpenCV-based medical-image operations for MedQCNN research.

Focused primitives that complement the existing preprocessing pipeline:

- ``clahe``         Contrast-Limited Adaptive Histogram Equalization.
                    Useful for low-contrast modalities (mammography,
                    chest X-ray, ultrasound).
- ``gamma``         Gamma correction for non-linear intensity remapping.
- ``bilateral``     Edge-preserving denoise (retains lesion boundaries).
- ``unsharp``       Unsharp-mask sharpening of fine structures.

All functions accept a 2-D single-channel image as ``np.ndarray`` with
values in ``[0, 1]`` (float) or ``[0, 255]`` (uint8) and return an array
of the same dtype.
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_u8(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (uint8 view, was_float) so we can restore the input dtype."""
    if image.dtype == np.uint8:
        return image, False
    arr = np.clip(image.astype(np.float32), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8), True


def _restore(image: np.ndarray, was_float: bool) -> np.ndarray:
    return (image.astype(np.float32) / 255.0) if was_float else image


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization."""
    u8, was_float = _to_u8(image)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return _restore(op.apply(u8), was_float)


def gamma(image: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Gamma correction. ``value < 1`` brightens; ``value > 1`` darkens."""
    if value <= 0:
        raise ValueError("gamma value must be positive")
    u8, was_float = _to_u8(image)
    lut = np.array(
        [((i / 255.0) ** (1.0 / value)) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    return _restore(cv2.LUT(u8, lut), was_float)


def bilateral(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 35.0,
    sigma_space: float = 7.0,
) -> np.ndarray:
    """Edge-preserving bilateral denoise."""
    u8, was_float = _to_u8(image)
    out = cv2.bilateralFilter(u8, diameter, sigma_color, sigma_space)
    return _restore(out, was_float)


def unsharp(
    image: np.ndarray,
    ksize: tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Unsharp-mask sharpening: ``image + amount * (image - blur)``."""
    u8, was_float = _to_u8(image)
    blur = cv2.GaussianBlur(u8, ksize, sigma)
    out = cv2.addWeighted(u8, 1.0 + amount, blur, -amount, 0)
    return _restore(out, was_float)
