"""
Device management for CPU/GPU compute.

Handles device detection and memory-aware selection for the
edge-compute constraints defined in GEMINI.md (16–32 GB RAM
on Raspberry Pi 5 cluster).
"""

from __future__ import annotations

import torch

from medqcnn.config.constants import SEED


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Detect and return the best available compute device.

    On edge hardware (Pi 5), this will always return 'cpu'.
    On development machines with CUDA, returns 'cuda'.

    Args:
        prefer_gpu: Whether to prefer GPU if available.

    Returns:
        torch.device instance.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Seeds: Python random, NumPy, PyTorch (CPU & CUDA).

    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_memory_info() -> dict[str, float]:
    """Get current memory usage in GB.

    Returns:
        Dict with 'ram_used_gb', 'ram_total_gb', and optionally
        'gpu_used_gb', 'gpu_total_gb'.
    """
    import psutil

    ram = psutil.virtual_memory()
    info: dict[str, float] = {
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_percent": ram.percent,
    }

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.mem_get_info()
        info["gpu_free_gb"] = round(gpu_mem[0] / (1024**3), 2)
        info["gpu_total_gb"] = round(gpu_mem[1] / (1024**3), 2)

    return info
