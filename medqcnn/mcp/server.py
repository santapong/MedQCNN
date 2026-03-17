"""
MedQCNN MCP Server — Model Context Protocol for AI Agent Integration.

Exposes the HybridQCNN model as tools that AI agents (e.g., LangChain,
Claude, etc.) can invoke for medical image diagnostics.

Tools:
  - diagnose:       Run inference on a medical image → diagnosis
  - model_info:     Get model architecture and parameter details
  - list_datasets:  List available MedMNIST benchmark datasets

Usage:
    uv run python scripts/mcp_server.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from mcp.server.fastmcp import FastMCP

from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed

# --- Initialize MCP Server ---
mcp = FastMCP("MedQCNN")

# --- Global model state ---
_model: HybridQCNN | None = None
_device: torch.device = torch.device("cpu")
_labels: list[str] = ["Benign", "Malignant"]


def _ensure_model(
    n_qubits: int = DEMO_QUBITS,
    n_layers: int = NUM_ANSATZ_LAYERS,
    checkpoint_path: str | None = None,
) -> HybridQCNN:
    """Lazy-load the model on first use."""
    global _model, _device, _labels

    if _model is not None:
        return _model

    set_seed()
    _device = get_device()

    # Peek at checkpoint to determine n_classes and labels
    n_classes = 2
    ckpt = None
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=_device)
        if "labels" in ckpt and ckpt["labels"] is not None:
            _labels = ckpt["labels"]
            n_classes = len(_labels)

    _model = HybridQCNN(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=n_classes,
        pretrained=True,
    ).to(_device)

    if ckpt is not None:
        _model.load_state_dict(ckpt["model_state_dict"])

    _model.eval()
    return _model


@mcp.tool()
def diagnose(image_path: str) -> str:
    """Analyze a medical image and return a diagnostic prediction.

    Runs the image through the full hybrid quantum-classical pipeline:
    ResNet backbone → Latent projection → Quantum circuit → Classification.

    Args:
        image_path: Absolute path to a medical image (PNG, JPG, DICOM).

    Returns:
        JSON string with prediction, confidence, probabilities, and
        quantum expectation values.
    """
    from PIL import Image
    from torchvision import transforms

    model = _ensure_model()

    # Load and preprocess image
    if not Path(image_path).exists():
        return json.dumps({"error": f"Image not found: {image_path}"})

    image = Image.open(image_path).convert("L")  # grayscale
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(_device)

    # Run inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()

        # Get quantum expectation values
        features = model.backbone(tensor)
        z = model.projector(features)
        q_out = model.quantum_layer(z)
        q_values = q_out[0].cpu().tolist()

    labels = _labels
    result = {
        "prediction": pred,
        "label": labels[pred] if pred < len(labels) else str(pred),
        "confidence": round(confidence, 6),
        "probabilities": {
            labels[i]: round(p, 6)
            for i, p in enumerate(probs[0].cpu().tolist())
            if i < len(labels)
        },
        "quantum_expectation_values": [round(v, 6) for v in q_values],
        "model_config": {
            "n_qubits": model.n_qubits,
            "latent_dim": 2**model.n_qubits,
            "ansatz_layers": model.n_layers,
        },
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def model_info() -> str:
    """Get the HybridQCNN model architecture and parameter details.

    Returns model configuration, trainable parameter counts by component
    (projector, quantum, classifier), and hardware info.

    Returns:
        JSON string with model architecture details.
    """
    model = _ensure_model()
    param_counts = model.count_trainable_params()

    info = {
        "model": "HybridQCNN",
        "version": "0.1.0",
        "architecture": {
            "n_qubits": model.n_qubits,
            "latent_dim": 2**model.n_qubits,
            "n_ansatz_layers": model.n_layers,
            "backbone": "ResNet-18 (frozen)",
            "quantum_gates": ["Ry", "Rz", "CZ"],
            "entanglement": "ring topology",
            "measurement": "local Pauli-Z per qubit",
        },
        "trainable_parameters": param_counts,
        "device": str(_device),
        "description": (
            "Hybrid quantum-classical model for medical image classification. "
            "Uses a frozen ResNet-18 backbone for feature extraction, projects "
            "to a quantum-compatible latent space, then evaluates via a "
            "parameterized quantum circuit with hardware-efficient ansatz."
        ),
    }
    return json.dumps(info, indent=2)


@mcp.tool()
def list_datasets() -> str:
    """List available MedMNIST benchmark datasets for training and evaluation.

    Returns dataset names, descriptions, number of classes, and sample counts.

    Returns:
        JSON string with available datasets.
    """
    datasets = {
        "breastmnist": {
            "description": "Breast ultrasound - benign vs malignant",
            "n_classes": 2,
            "task": "binary classification",
            "image_size": "28x28",
        },
        "pathmnist": {
            "description": "Colon pathology - 9 tissue types",
            "n_classes": 9,
            "task": "multi-class classification",
            "image_size": "28x28",
        },
        "dermamnist": {
            "description": "Dermatoscopy - 7 skin lesion types",
            "n_classes": 7,
            "task": "multi-class classification",
            "image_size": "28x28",
        },
        "organamnist": {
            "description": "Abdominal CT - 11 organ types (axial)",
            "n_classes": 11,
            "task": "multi-class classification",
            "image_size": "28x28",
        },
        "bloodmnist": {
            "description": "Blood cell microscope - 8 cell types",
            "n_classes": 8,
            "task": "multi-class classification",
            "image_size": "28x28",
        },
        "custom": {
            "description": "Custom image dataset in ImageFolder format (train/val/test with class subdirectories)",
            "n_classes": "auto-detected from directory structure",
            "task": "classification",
            "image_size": "user-defined",
            "usage": "Use --dataset custom --data-dir /path/to/data with the training script",
        },
    }
    return json.dumps(datasets, indent=2)


def run_server() -> None:
    """Start the MCP server (stdio transport)."""
    mcp.run()
