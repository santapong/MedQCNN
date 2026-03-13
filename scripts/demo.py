"""
MedQCNN Demo: End-to-End Forward Pass

Demonstrates the full hybrid quantum-classical pipeline:
  1. Download a small MedMNIST dataset (BreastMNIST)
  2. Run one batch through the HybridQCNN model
  3. Print shapes, quantum outputs, predictions, and param counts

Usage:
    uv run python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from medqcnn.config.constants import DEMO_QUBITS, NUM_ANSATZ_LAYERS
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.utils.device import get_device, set_seed
from medqcnn.utils.logging import console


def run_demo() -> None:
    """Run end-to-end demo of the hybrid pipeline."""
    console.rule("[bold cyan]MedQCNN Demo: End-to-End Forward Pass[/bold cyan]")

    # --- Setup ---
    set_seed()
    device = get_device(prefer_gpu=False)  # CPU for demo reliability
    n_qubits = DEMO_QUBITS  # 4 qubits → 16-dim latent (fast!)
    latent_dim = 2**n_qubits  # 16

    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Qubits:        {n_qubits}")
    console.print(f"  Latent dim:    {latent_dim}")
    console.print(f"  Ansatz layers: {NUM_ANSATZ_LAYERS}")
    console.print(f"  Device:        {device}")

    # --- Step 1: Load a small MedMNIST dataset ---
    console.print("\n[bold yellow]Step 1:[/bold yellow] Loading BreastMNIST...")
    from medqcnn.data.loader import get_medmnist_loaders

    train_loader, val_loader, test_loader = get_medmnist_loaders(
        dataset_name="breastmnist",
        batch_size=4,  # Small batch for demo
        data_dir="data",
        download=True,
    )
    console.print(f"  Train samples: {len(train_loader.dataset)}")
    console.print(f"  Val samples:   {len(val_loader.dataset)}")
    console.print(f"  Test samples:  {len(test_loader.dataset)}")

    # --- Step 2: Build the model ---
    console.print("\n[bold yellow]Step 2:[/bold yellow] Building HybridQCNN...")
    model = HybridQCNN(
        n_qubits=n_qubits,
        n_layers=NUM_ANSATZ_LAYERS,
        n_classes=2,
        pretrained=False,  # Avoid downloading weights for demo
    )
    model = model.to(device)
    model.eval()

    param_counts = model.count_trainable_params()
    console.print("  [green]Trainable parameters:[/green]")
    for name, count in param_counts.items():
        console.print(f"    {name:>12s}: {count:,}")

    # --- Step 3: Run one batch ---
    console.print("\n[bold yellow]Step 3:[/bold yellow] Running forward pass...")
    images, labels = next(iter(test_loader))

    # MedMNIST images are 28x28 — resize to 224x224 for ResNet
    images_resized = torch.nn.functional.interpolate(
        images.float(), size=(224, 224), mode="bilinear", align_corners=False
    )
    images_resized = images_resized.to(device)
    labels = labels.squeeze()

    console.print(f"  Input shape:   {list(images_resized.shape)}")

    with torch.no_grad():
        logits = model(images_resized)

    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    console.print(f"  Output logits: {logits.detach().cpu().tolist()}")
    console.print(f"  Probabilities: {probs.detach().cpu().tolist()}")
    console.print(f"  Predictions:   {preds.detach().cpu().tolist()}")
    console.print(f"  True labels:   {labels.tolist()}")

    # --- Step 4: Verify gradient flow ---
    console.print("\n[bold yellow]Step 4:[/bold yellow] Verifying gradient flow...")
    model.train()
    images_grad = images_resized[:2].clone().requires_grad_(False)
    labels_grad = labels[:2].long()

    logits_grad = model(images_grad)
    loss = torch.nn.functional.cross_entropy(logits_grad, labels_grad)
    loss.backward()

    # Check quantum layer has gradients
    has_quantum_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.quantum_layer.parameters()
    )
    has_projector_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.projector.parameters()
    )

    if has_quantum_grad:
        console.print("  [bold green]✓[/bold green] Quantum parameters have gradients!")
    else:
        console.print("  [bold red]✗[/bold red] Quantum parameters have NO gradients")

    if has_projector_grad:
        console.print(
            "  [bold green]✓[/bold green] Projector parameters have gradients!"
        )
    else:
        console.print("  [bold red]✗[/bold red] Projector parameters have NO gradients")

    console.print(f"  Loss value:    {loss.item():.4f}")

    console.rule("[bold green]Demo Complete[/bold green]")


if __name__ == "__main__":
    run_demo()
