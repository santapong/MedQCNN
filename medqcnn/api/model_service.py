"""Encapsulated model state and inference logic for the MedQCNN API."""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
from pathlib import Path

import torch
from litestar.exceptions import ClientException

from medqcnn.api.schemas import PredictionResponse
from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.inference.conformal import ConformalPredictor
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.training.calibration import apply_temperature
from medqcnn.utils.device import get_device, set_seed

logger = logging.getLogger("medqcnn.api")


class ModelService:
    """Manages model lifecycle and inference.

    Encapsulates the previously global model state (_model, _device, _labels)
    into a single class for better testability and thread-safety.

    Optional sidecar files loaded next to the checkpoint:

    * ``<ckpt>.calibration.json`` — ``{"temperature": float}`` from
      :py:mod:`medqcnn.training.calibration`.
    * ``<ckpt>.conformal.json`` — serialised
      :class:`~medqcnn.inference.conformal.ConformalPredictor`.

    Their presence is silently optional: when missing, the response
    fields ``calibrated_confidence`` / ``prediction_set`` / ``abstained``
    are returned as ``None`` and behaviour matches the pre-trust-layer API.
    """

    def __init__(self) -> None:
        self.model: HybridQCNN | None = None
        self.device: torch.device = torch.device("cpu")
        self.labels: list[str] = ["Benign", "Malignant"]
        self.temperature: float | None = None
        self.conformal: ConformalPredictor | None = None

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
            self._load_trust_sidecars(Path(checkpoint_path))
        else:
            self.temperature = None
            self.conformal = None

        self.model.eval()

    def _load_trust_sidecars(self, checkpoint_path: Path) -> None:
        """Pick up temperature / conformal sidecars next to the checkpoint."""
        suffix = checkpoint_path.suffix
        calib_path = checkpoint_path.with_suffix(suffix + ".calibration.json")
        if calib_path.exists():
            try:
                payload = json.loads(calib_path.read_text())
                self.temperature = float(payload["temperature"])
                logger.info(
                    "Loaded temperature scaling T=%.4f from %s",
                    self.temperature,
                    calib_path.name,
                )
            except (ValueError, KeyError, OSError):
                logger.warning(
                    "Failed to load calibration sidecar %s",
                    calib_path,
                    exc_info=True,
                )
                self.temperature = None
        else:
            self.temperature = None

        conf_path = checkpoint_path.with_suffix(suffix + ".conformal.json")
        if conf_path.exists():
            try:
                self.conformal = ConformalPredictor.load(conf_path)
                logger.info(
                    "Loaded conformal predictor (alpha=%.2f, qhat=%.4f) from %s",
                    self.conformal.alpha,
                    self.conformal.qhat or float("nan"),
                    conf_path.name,
                )
            except (ValueError, KeyError, OSError):
                logger.warning(
                    "Failed to load conformal sidecar %s",
                    conf_path,
                    exc_info=True,
                )
                self.conformal = None
        else:
            self.conformal = None

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

            calibrated_confidence: float | None = None
            calibrated_probs: torch.Tensor = probs
            if self.temperature is not None:
                calibrated_probs = torch.softmax(
                    apply_temperature(logits, self.temperature), dim=1
                )
                calibrated_confidence = round(
                    float(calibrated_probs[0, pred].item()), 6
                )

        q_values = None
        if hasattr(self.model, "quantum_layer"):
            with torch.no_grad():
                features = self.model.backbone(tensor)
                z = self.model.projector(features)
                q_out = self.model.quantum_layer(z)
                q_values = q_out[0].cpu().tolist()

        prediction_set: list[int] | None = None
        prediction_set_labels: list[str] | None = None
        abstained: bool | None = None
        if self.conformal is not None and self.conformal.is_fitted:
            sets = self.conformal.predict_set(calibrated_probs)
            prediction_set = sets[0]
            prediction_set_labels = [
                self.labels[i] if i < len(self.labels) else str(i)
                for i in prediction_set
            ]
            abstained = len(prediction_set) != 1

        return PredictionResponse(
            prediction=pred,
            label=self.labels[pred] if pred < len(self.labels) else str(pred),
            confidence=round(confidence, 6),
            probabilities=probs[0].cpu().tolist(),
            quantum_expectation_values=q_values,
            calibrated_confidence=calibrated_confidence,
            prediction_set=prediction_set,
            prediction_set_labels=prediction_set_labels,
            abstained=abstained,
        )


# Module-level singleton
model_service = ModelService()
