"""
MedQCNN Training Script

CLI-based training using the HybridQCNN model and Trainer class.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 10 --n-qubits 4 --batch-size 8
    uv run python scripts/train.py --dataset pathmnist --lr 0.0005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from medqcnn.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    NUM_ANSATZ_LAYERS,
    NUM_QUBITS,
)
from medqcnn.data.loader import get_medmnist_loaders
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.training.trainer import Trainer
from medqcnn.utils.device import get_device, get_memory_info, set_seed
from medqcnn.utils.logging import console, setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the MedQCNN Hybrid Quantum-Classical Model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breastmnist",
        help="Dataset name: MedMNIST name or 'custom' (requires --data-dir)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to custom dataset directory (with train/val/test subdirs)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=NUM_QUBITS,
        help=f"Number of qubits (default: {NUM_QUBITS})",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=NUM_ANSATZ_LAYERS,
        help=f"Number of ansatz layers (default: {NUM_ANSATZ_LAYERS})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (0 to disable, default: 10)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    console.rule("[bold cyan]MedQCNN Training[/bold cyan]")
    console.print(f"  Dataset:       {args.dataset}")
    console.print(f"  Epochs:        {args.epochs}")
    console.print(f"  Batch size:    {args.batch_size}")
    console.print(f"  Learning rate: {args.lr}")
    console.print(f"  Qubits:        {args.n_qubits}")
    console.print(f"  Ansatz layers: {args.n_layers}")
    console.print(f"  Latent dim:    {2**args.n_qubits}")

    # Setup
    set_seed(args.seed)
    device = get_device()
    mem = get_memory_info()
    console.print(
        f"  Device:        {device} | "
        f"RAM: {mem['ram_used_gb']:.1f}/{mem['ram_total_gb']:.1f} GB"
    )

    # Data
    console.print(f"\n[bold yellow]Loading {args.dataset}...[/bold yellow]")

    if args.dataset == "custom":
        if not args.data_dir:
            console.print("[bold red]--data-dir is required for custom datasets[/bold red]")
            sys.exit(1)
        from medqcnn.data.loader import get_custom_loaders

        train_loader, val_loader, _, label_names = get_custom_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
        )
        n_classes = len(label_names)
        use_resized_loader = False
    else:
        train_loader, val_loader, _ = get_medmnist_loaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            download=True,
        )
        from medmnist import INFO

        info = INFO[args.dataset]
        n_classes = len(info["label"])
        label_names = [info["label"][str(i)] for i in range(n_classes)]
        use_resized_loader = True

    console.print(
        f"  Train: {len(train_loader.dataset)} samples | "
        f"Val: {len(val_loader.dataset)} samples"
    )
    console.print(f"  Classes ({n_classes}): {label_names}")

    # Model
    console.print("\n[bold yellow]Building HybridQCNN...[/bold yellow]")
    model = HybridQCNN(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        pretrained=True,
    )

    param_counts = model.count_trainable_params()
    console.print(f"  Trainable params: {param_counts['total']:,}")
    for name, count in param_counts.items():
        if name != "total":
            console.print(f"    {name:>12s}: {count:,}")

    # Wrap DataLoaders to resize images to 224x224 (only for MedMNIST 28x28 images)
    if use_resized_loader:

        class ResizedLoader:
            """Wrapper that resizes MedMNIST images to ResNet input size."""

            def __init__(self, loader):
                self.loader = loader
                self.dataset = loader.dataset

            def __iter__(self):
                for images, labels in self.loader:
                    images = torch.nn.functional.interpolate(
                        images.float(),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    yield images, labels

            def __len__(self):
                return len(self.loader)

        resized_train = ResizedLoader(train_loader)
        resized_val = ResizedLoader(val_loader)
    else:
        resized_train = train_loader
        resized_val = val_loader

    # Train
    console.rule("[bold yellow]Training[/bold yellow]")
    trainer = Trainer(
        model=model,
        train_loader=resized_train,
        val_loader=resized_val,
        learning_rate=args.lr,
        device=str(device),
        dataset_name=args.dataset,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        labels=label_names,
    )

    # Resume from checkpoint if specified
    resume_from = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_from = trainer.load_checkpoint(resume_path)
            console.print(f"  Resumed from checkpoint: {args.resume} (epoch {resume_from})")
        else:
            console.print(f"[bold red]Checkpoint not found: {args.resume}[/bold red]")
            sys.exit(1)

    history = trainer.fit(
        epochs=args.epochs,
        save_every=args.save_every,
        patience=args.patience,
        resume_from=resume_from,
    )

    # Summary
    console.rule("[bold green]Training Complete[/bold green]")
    final_train_acc = history["train_acc"][-1]
    final_val_acc = history["val_acc"][-1]
    console.print(f"  Final Train Acc: {final_train_acc:.4f}")
    console.print(f"  Final Val Acc:   {final_val_acc:.4f}")
    console.print("  Checkpoints saved to: checkpoints/")


if __name__ == "__main__":
    main()
