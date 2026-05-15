"""
Noise sweep harness: train HybridQCNN under each Aer noise preset for
multiple seeds and persist the results to the database.

Per the project's research plan (research.md §6 #1), this is the
top-listed open item. The sweep is the smallest experiment that
converts MedQCNN from "ideal simulator only" to "NISQ-credible".

Usage:
    # Smoke test: 1 epoch, 1 seed, all 4 presets
    uv run python scripts/noise_sweep.py --smoke

    # Default research run (3 seeds × 4 presets × 50 epochs)
    uv run python scripts/noise_sweep.py --dataset breastmnist

    # Custom
    uv run python scripts/noise_sweep.py \
        --dataset pneumoniamnist --epochs 20 --seeds 42 43 44 \
        --presets none low medium

Each completed run writes a `training_runs` row whose
`noise_config_json` column records the depolarising/readout rates,
so the dashboard can plot accuracy vs. noise without re-parsing logs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from medqcnn.config.constants import DEFAULT_BATCH_SIZE, NUM_ANSATZ_LAYERS, NUM_QUBITS
from medqcnn.data.loader import get_medmnist_loaders
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.quantum.noise import NOISE_PRESETS, get_preset
from medqcnn.training.trainer import Trainer
from medqcnn.utils.device import get_device, set_seed
from medqcnn.utils.logging import console


def _resized_loader(loader):
    """Wrap a 28x28 MedMNIST loader to upsample to 224x224."""

    class _R:
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

    return _R(loader)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Noise sweep for HybridQCNN")
    p.add_argument("--dataset", type=str, default="breastmnist")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--n-qubits", type=int, default=NUM_QUBITS)
    p.add_argument("--n-layers", type=int, default=NUM_ANSATZ_LAYERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds to sweep (default: 42 43 44).",
    )
    p.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=sorted(NOISE_PRESETS),
        choices=sorted(NOISE_PRESETS),
        help="Noise presets to sweep (default: all four).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: override to 1 epoch × 1 seed × all 4 presets.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.seeds = [args.seeds[0]]

    device = str(get_device())
    console.rule("[bold cyan]MedQCNN Noise Sweep[/bold cyan]")
    console.print(f"  Dataset:  {args.dataset}")
    console.print(f"  Epochs:   {args.epochs}")
    console.print(f"  Seeds:    {args.seeds}")
    console.print(f"  Presets:  {args.presets}")
    console.print(f"  Qubits:   {args.n_qubits}")
    console.print(f"  Device:   {device}")

    # MedMNIST loaders are seed-independent at the dataset level; we
    # only re-seed the model + sampler before each run.
    train_loader, val_loader, _ = get_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        download=True,
    )
    from medmnist import INFO

    info = INFO[args.dataset]
    n_classes = len(info["label"])
    label_names = [info["label"][str(i)] for i in range(n_classes)]

    train_loader = _resized_loader(train_loader)
    val_loader = _resized_loader(val_loader)

    overall_start = time.time()
    completed: list[tuple[str, int, float]] = []

    for preset in args.presets:
        noise_config = get_preset(preset)
        for seed in args.seeds:
            run_label = f"{preset} / seed={seed}"
            console.rule(f"[bold yellow]{run_label}[/bold yellow]")
            set_seed(seed)

            model = HybridQCNN(
                n_qubits=args.n_qubits,
                n_layers=args.n_layers,
                n_classes=n_classes,
                pretrained=True,
                noise_config=noise_config if not noise_config.is_noiseless else None,
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=args.lr,
                device=device,
                dataset_name=args.dataset,
                n_qubits=args.n_qubits,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                labels=label_names,
                noise_config=noise_config if not noise_config.is_noiseless else None,
            )

            run_start = time.time()
            history = trainer.fit(epochs=args.epochs, save_every=args.epochs)
            duration = time.time() - run_start

            final_val = history["val_acc"][-1] if history["val_acc"] else float("nan")
            console.print(f"  {run_label}: val_acc={final_val:.4f}  ({duration:.1f}s)")
            completed.append((preset, seed, final_val))

    total = time.time() - overall_start
    console.rule("[bold green]Sweep Complete[/bold green]")
    console.print(f"  Total runs:     {len(completed)}")
    console.print(f"  Total duration: {total / 60:.1f} min")
    console.print("  Final val accuracy by run:")
    for preset, seed, acc in completed:
        console.print(f"    {preset:>7s} seed={seed}: {acc:.4f}")


if __name__ == "__main__":
    main()
