"""
MedQCNN Evaluation Script

Evaluate a trained HybridQCNN model on the test set.
Computes accuracy, AUC-ROC, F1, and generates visualizations.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --checkpoint checkpoints/model_final.pt
    uv run python scripts/evaluate.py --n-qubits 4 --dataset breastmnist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from medqcnn.config.constants import (
    DEFAULT_BATCH_SIZE,
    NUM_ANSATZ_LAYERS,
    NUM_QUBITS,
)
from medqcnn.data.loader import get_medmnist_loaders
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.training.metrics import compute_accuracy, compute_auc_roc, compute_f1
from medqcnn.training.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
)
from medqcnn.utils.device import get_device, set_seed
from medqcnn.utils.logging import console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MedQCNN model")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument("--dataset", type=str, default="breastmnist")
    parser.add_argument("--n-qubits", type=int, default=NUM_QUBITS)
    parser.add_argument("--n-layers", type=int, default=NUM_ANSATZ_LAYERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Run evaluation and return metrics."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = torch.nn.functional.interpolate(
            images.float(), size=(224, 224), mode="bilinear", align_corners=False
        ).to(device)
        labels = labels.squeeze()

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(
            probs[:, 1].cpu().numpy().tolist()
            if probs.shape[1] == 2
            else probs.cpu().numpy().tolist()
        )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    accuracy = compute_accuracy(y_true, y_pred)
    f1 = compute_f1(y_true, y_pred, average="weighted")

    results = {
        "accuracy": accuracy,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }

    # AUC-ROC only for binary classification
    if len(np.unique(y_true)) == 2:
        results["auc_roc"] = compute_auc_roc(y_true, y_score)

    return results


def main() -> None:
    args = parse_args()

    console.rule("[bold cyan]MedQCNN Evaluation[/bold cyan]")
    set_seed()
    device = get_device()

    # Load data
    _, _, test_loader = get_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        download=True,
    )

    from medmnist import INFO

    info = INFO[args.dataset]
    n_classes = len(info["label"])

    # Build model
    model = HybridQCNN(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        pretrained=True,
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print(f"  Loaded checkpoint: {args.checkpoint}")

        # Plot training history if available
        if "history" in checkpoint:
            plot_training_history(checkpoint["history"])
            console.print("  Saved: docs/training_history.png")
    else:
        console.print("  [yellow]No checkpoint — evaluating untrained model[/yellow]")

    # Evaluate
    console.print(f"\n  Evaluating on {len(test_loader.dataset)} test samples...")
    results = evaluate(model, test_loader, device)

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"  Accuracy:  {results['accuracy']:.4f}")
    console.print(f"  F1 Score:  {results['f1']:.4f}")

    if "auc_roc" in results:
        console.print(f"  AUC-ROC:   {results['auc_roc']:.4f}")

    # Generate visualizations
    class_names = (
        list(info["label"].values())
        if isinstance(info["label"], dict)
        else [str(i) for i in range(n_classes)]
    )
    plot_confusion_matrix(results["y_true"], results["y_pred"], class_names=class_names)
    console.print("  Saved: docs/confusion_matrix.png")

    if "auc_roc" in results:
        plot_roc_curve(results["y_true"], results["y_score"])
        console.print("  Saved: docs/roc_curve.png")

    console.rule("[bold green]Evaluation Complete[/bold green]")


if __name__ == "__main__":
    main()
