"""Encapsulated model state and inference logic for the MedQCNN API."""

from __future__ import annotations

import base64
import binascii
import io
import logging
from pathlib import Path

import torch
from litestar.exceptions import ClientException

from medqcnn.api.schemas import PredictionResponse
from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed

logger = logging.getLogger("medqcnn.api")


class ModelService:
    """Manages model lifecycle and inference.

    Encapsulates the previously global model state (_model, _device, _labels)
    into a single class for better testability and thread-safety.
    """

    def __init__(self) -> None:
        self.model: HybridQCNN | None = None
        self.device: torch.device = torch.device("cpu")
        self.labels: list[str] = ["Benign", "Malignant"]

    def load(
        self,
        checkpoint_path: str | None = None,
        n_qubits: int = DEMO_QUBITS,
        n_layers: int = NUM_ANSATZ_LAYERS,
        n_classes: int = 2,
    ) -> None:
        """Load the HybridQCNN model."""
        set_seed()
        self.device = get_device()

        self.model = HybridQCNN(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=n_classes,
            pretrained=True,
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Load dynamic labels from checkpoint (fallback to default)
            if "labels" in checkpoint and checkpoint["labels"] is not None:
                self.labels = checkpoint["labels"]
                # Rebuild model if n_classes differs from checkpoint labels
                if n_classes != len(self.labels):
                    n_classes = len(self.labels)
                    self.model = HybridQCNN(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        n_classes=n_classes,
                        pretrained=True,
                    ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

    def ensure_loaded(self) -> None:
        """Lazy-load model if not already loaded."""
        if self.model is None:
            self.load()

    def run_inference(self, image_base64: str) -> PredictionResponse:
        """Run inference on a base64-encoded image."""
        from PIL import Image, UnidentifiedImageError
        from torchvision import transforms

        self.ensure_loaded()

        # Validate base64 payload size (max 250 MB encoded)
        max_payload_bytes = 250 * 1024 * 1024
        if len(image_base64) > max_payload_bytes:
            raise ClientException(
                detail=f"Image payload too large. Max {max_payload_bytes} bytes.",
                status_code=413,
            )

        try:
            image_bytes = base64.b64decode(image_base64)
        except binascii.Error:
            raise ClientException(
                detail="Invalid base64 encoding.",
                status_code=400,
            )

        # Validate decoded image size (max 50 MB)
        max_decoded_bytes = 50 * 1024 * 1024
        if len(image_bytes) > max_decoded_bytes:
            raise ClientException(
                detail=(
                    f"Decoded image too large ({len(image_bytes)} bytes). "
                    f"Max {max_decoded_bytes} bytes."
                ),
                status_code=413,
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except (UnidentifiedImageError, OSError):
            raise ClientException(
                detail="Cannot decode image. Supported formats: PNG, JPEG, BMP, TIFF.",
                status_code=400,
            )

        max_dim = 4096
        if image.width > max_dim or image.height > max_dim:
            raise ClientException(
                detail=(
                    f"Image dimensions too large ({image.width}x{image.height}). "
                    f"Max {max_dim}x{max_dim}."
                ),
                status_code=400,
            )

        image = image.convert("L")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        q_values = None
        if hasattr(self.model, "quantum_layer"):
            with torch.no_grad():
                features = self.model.backbone(tensor)
                z = self.model.projector(features)
                q_out = self.model.quantum_layer(z)
                q_values = q_out[0].cpu().tolist()

        return PredictionResponse(
            prediction=pred,
            label=self.labels[pred] if pred < len(self.labels) else str(pred),
            confidence=round(confidence, 6),
            probabilities=probs[0].cpu().tolist(),
            quantum_expectation_values=q_values,
        )


# Module-level singleton
model_service = ModelService()
