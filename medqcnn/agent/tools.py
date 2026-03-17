"""
LangChain Tool Wrappers for MedQCNN.

Exposes the quantum-classical inference pipeline as LangChain-compatible
tools that can be used by any LangChain agent (ReAct, OpenAI Functions,
LangGraph, etc.) for the CaaS-Q agentic AI network.

Tools:
  - quantum_diagnose:  Run quantum-classical inference on medical images
  - get_model_info:    Retrieve model architecture details
  - list_datasets:     List available benchmark datasets
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from langchain_core.tools import tool

from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed

# --- Lazy model singleton ---
_model: HybridQCNN | None = None
_device: torch.device = torch.device("cpu")
_labels: list[str] = ["Benign", "Malignant"]


def _get_model(
    n_qubits: int = DEMO_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    checkpoint: str | None = None,
) -> HybridQCNN:
    """Lazy-load the HybridQCNN model."""
    global _model, _device, _labels

    if _model is not None:
        return _model

    set_seed()
    _device = get_device()

    # Peek at checkpoint to determine n_classes and labels
    n_classes = 2
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=_device)
        if "labels" in ckpt and ckpt["labels"] is not None:
            _labels = ckpt["labels"]
            n_classes = len(_labels)

    _model = HybridQCNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        pretrained=True,
    ).to(_device)

    if checkpoint and Path(checkpoint).exists():
        _model.load_state_dict(ckpt["model_state_dict"])

    _model.eval()
    return _model


@tool
def quantum_diagnose(image_path: str) -> str:
    """Run quantum-classical medical image diagnosis.

    Analyzes a medical image through the full HybridQCNN pipeline:
    ResNet-18 feature extraction → Latent projection to R^16 →
    4-qubit quantum circuit with amplitude encoding and HEA ansatz →
    Per-qubit Pauli-Z measurement → Classification.

    Args:
        image_path: Absolute path to a medical image file (PNG, JPG, BMP).

    Returns:
        JSON with prediction (Benign/Malignant), confidence score,
        class probabilities, and quantum expectation values.
    """
    from PIL import Image
    from torchvision import transforms

    model = _get_model()
    path = Path(image_path)

    if not path.exists():
        return json.dumps({"error": f"File not found: {image_path}"})

    # Load and preprocess
    image = Image.open(path).convert("L")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(_device)

    # Quantum-classical inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

        # Quantum expectation values
        features = model.backbone(tensor)
        z = model.projector(features)
        q_out = model.quantum_layer(z)
        q_values = q_out[0].cpu().tolist()

    labels = _labels
    return json.dumps(
        {
            "prediction": labels[pred] if pred < len(labels) else str(pred),
            "prediction_code": pred,
            "confidence": round(confidence, 4),
            "probabilities": {
                labels[i]: round(probs[0, i].item(), 4)
                for i in range(min(len(labels), probs.shape[1]))
            },
            "quantum_expectation_values": [round(v, 4) for v in q_values],
            "model": {
                "type": "HybridQCNN",
                "qubits": model.n_qubits,
                "ansatz_layers": model.n_layers,
            },
        },
        indent=2,
    )


@tool
def get_model_info() -> str:
    """Get the HybridQCNN model architecture and parameter details.

    Returns architecture configuration, trainable parameter counts
    for each component (projector, quantum circuit, classifier),
    and deployment information.
    """
    model = _get_model()
    params = model.count_trainable_params()

    return json.dumps(
        {
            "model": "HybridQCNN v0.1.0",
            "architecture": {
                "backbone": "ResNet-18 (frozen, pre-trained on ImageNet)",
                "projector": f"FC → R^{2**model.n_qubits} with BatchNorm + L2 norm",
                "quantum_circuit": {
                    "qubits": model.n_qubits,
                    "encoding": "Amplitude encoding (data → quantum state)",
                    "ansatz": f"Hardware-Efficient Ansatz ({model.n_layers} layers)",
                    "gates": "Ry, Rz rotations + CZ entanglement (ring topology)",
                    "measurement": "Per-qubit Pauli-Z expectation values",
                },
                "classifier": f"FC {model.n_qubits} → 32 → ReLU → Dropout → n_classes",
            },
            "trainable_parameters": params,
            "quantum_advantage": (
                f"Only {params['quantum']} quantum parameters operate in a "
                f"{2**model.n_qubits}-dimensional Hilbert space, achieving "
                f"exponential compression vs classical FC layers."
            ),
        },
        indent=2,
    )


@tool
def list_medical_datasets() -> str:
    """List available MedMNIST benchmark datasets for medical imaging.

    Returns supported datasets with descriptions, number of classes,
    and clinical applications.
    """
    return json.dumps(
        {
            "breastmnist": {
                "description": "Breast ultrasound images",
                "classes": ["Benign", "Malignant"],
                "n_classes": 2,
                "clinical_use": "Breast cancer screening",
            },
            "pathmnist": {
                "description": "Colon pathology tissue slides",
                "n_classes": 9,
                "clinical_use": "Colorectal cancer tissue classification",
            },
            "dermamnist": {
                "description": "Dermatoscopy skin lesion images",
                "n_classes": 7,
                "clinical_use": "Skin cancer detection",
            },
            "bloodmnist": {
                "description": "Blood cell microscopy images",
                "n_classes": 8,
                "clinical_use": "Blood cell type classification",
            },
            "organamnist": {
                "description": "Abdominal CT organ scans (axial)",
                "n_classes": 11,
                "clinical_use": "Organ identification in CT scans",
            },
            "custom": {
                "description": "Custom image dataset in ImageFolder format",
                "n_classes": "auto-detected from directory structure",
                "clinical_use": "User-defined",
                "usage": "Use --dataset custom --data-dir /path/to/data with the training script",
            },
        },
        indent=2,
    )


# Convenience: list all tools for agent construction
ALL_TOOLS = [quantum_diagnose, get_model_info, list_medical_datasets]
