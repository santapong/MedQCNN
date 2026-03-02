"""
Classical CNN backbone for spatial feature extraction.

Implements a frozen pre-trained ResNet that acts as a deterministic
feature extractor (Node A in the pipeline). The backbone weights
are never updated during quantum-hybrid training — only the
projection head and quantum parameters are trainable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from medqcnn.config.constants import BACKBONE_NAME


class ClassicalBackbone(nn.Module):
    """Frozen ResNet feature extractor.

    Extracts spatial features from medical image slices and outputs
    a fixed-size feature vector for downstream projection to R^256.

    Args:
        backbone_name: Name of the torchvision model to use.
        pretrained: Whether to load ImageNet pre-trained weights.
        freeze: Whether to freeze all backbone parameters.
    """

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        # Load pre-trained ResNet
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)

        # Strip the final FC classifier — we only need the feature maps
        self.feature_dim = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze backbone if requested
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W).
               For grayscale medical images (C=1), the channel is
               replicated to 3 for ResNet compatibility.

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        # Replicate grayscale → 3-channel for ResNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.features(x)
        return features.flatten(start_dim=1)  # (B, feature_dim)
