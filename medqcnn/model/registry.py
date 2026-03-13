"""
Model version registry for MedQCNN.

Scans the checkpoints directory for .pt files and manages
loading/caching multiple model versions with LRU eviction.
Designed for edge deployment with limited RAM.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from medqcnn.config.constants import (
    CHECKPOINT_DIR,
    DEMO_QUBITS,
    MAX_CACHED_MODELS,
    NUM_ANSATZ_LAYERS,
)
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed

logger = logging.getLogger("medqcnn.registry")

# Pattern to extract version from filename: model_v1.pt, model_v2.pt, etc.
VERSION_PATTERN = re.compile(r"model_v(\d+)\.pt$")


@dataclass
class ModelVersion:
    """Metadata for a model checkpoint."""

    version: str
    path: str
    size_bytes: int
    modified: float

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "path": self.path,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
        }


class ModelRegistry:
    """LRU-cached model registry for multiple checkpoint versions."""

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        max_cached: int = MAX_CACHED_MODELS,
        n_qubits: int = DEMO_QUBITS,
        n_layers: int = NUM_ANSATZ_LAYERS,
        n_classes: int = 2,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_cached = max_cached
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self._device = get_device()

        # LRU cache: OrderedDict preserves insertion order
        self._cache: OrderedDict[str, HybridQCNN] = OrderedDict()
        self._versions: dict[str, ModelVersion] = {}
        self._active_version: str | None = None

        self.scan()

    def scan(self) -> list[ModelVersion]:
        """Scan checkpoint directory for available model files."""
        self._versions.clear()

        if not self.checkpoint_dir.exists():
            logger.info("Checkpoint directory %s does not exist", self.checkpoint_dir)
            return []

        for pt_file in sorted(self.checkpoint_dir.glob("*.pt")):
            version = self._extract_version(pt_file)
            stat = pt_file.stat()
            self._versions[version] = ModelVersion(
                version=version,
                path=str(pt_file),
                size_bytes=stat.st_size,
                modified=stat.st_mtime,
            )

        # Default active version = latest by modification time
        if self._versions and self._active_version is None:
            latest = max(self._versions.values(), key=lambda v: v.modified)
            self._active_version = latest.version

        logger.info(
            "Found %d checkpoint(s), active: %s",
            len(self._versions),
            self._active_version,
        )
        return list(self._versions.values())

    def _extract_version(self, path: Path) -> str:
        """Extract version string from checkpoint filename."""
        match = VERSION_PATTERN.search(path.name)
        if match:
            return f"v{match.group(1)}"
        # Fallback: use filename stem
        return path.stem

    def list_versions(self) -> list[ModelVersion]:
        """Return all available model versions."""
        return list(self._versions.values())

    @property
    def active_version(self) -> str | None:
        return self._active_version

    def set_active_version(self, version: str) -> None:
        """Set the active model version."""
        if version not in self._versions:
            raise ValueError(
                f"Version '{version}' not found. Available: {list(self._versions)}"
            )
        self._active_version = version

    def get_model(self, version: str | None = None) -> HybridQCNN:
        """Load and return a model by version. Uses LRU cache."""
        version = version or self._active_version
        if version is None:
            # No checkpoints — return fresh model
            set_seed()
            model = HybridQCNN(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_classes=self.n_classes,
                pretrained=True,
            ).to(self._device)
            model.eval()
            return model

        # Check cache
        if version in self._cache:
            self._cache.move_to_end(version)
            return self._cache[version]

        # Load from disk
        if version not in self._versions:
            raise ValueError(f"Version '{version}' not found")

        model_info = self._versions[version]
        set_seed()
        model = HybridQCNN(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_classes=self.n_classes,
            pretrained=True,
        ).to(self._device)

        checkpoint = torch.load(model_info.path, map_location=self._device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Cache with LRU eviction
        self._cache[version] = model
        if len(self._cache) > self.max_cached:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.info("Evicted model %s from cache", evicted_key)

        return model
