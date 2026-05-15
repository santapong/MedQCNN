"""
Eval-only noise sensitivity sweep (Idea 3 backend).

Loads a trained checkpoint, evaluates the model under each Aer noise
preset, and writes one `benchmarks` row per preset (with the noise
rates attached). The companion endpoint `/benchmarks/noise-sensitivity`
returns these rows for plotting.

Usage:
    uv run python scripts/noise_sensitivity_eval.py \
        --checkpoint checkpoints/model_final.pt \
        --training-run-id 7 \
        --dataset breastmnist
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
from medqcnn.utils.device import get_device
from medqcnn.utils.logging import console


def _resized_iter(loader):
    """28x28 → 224x224 wrapper for MedMNIST."""
    for images, labels in loader:
        images = torch.nn.functional.interpolate(
            images.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        yield images, labels


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in _resized_iter(loader):
        images = images.to(device)
        labels = labels.to(device).squeeze()
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval-only noise sensitivity sweep")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--training-run-id",
        type=int,
        default=None,
        help="DB id of the training run that produced this checkpoint. "
        "When given, sweep results are persisted as benchmarks.",
    )
    p.add_argument("--dataset", type=str, default="breastmnist")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--n-qubits", type=int, default=NUM_QUBITS)
    p.add_argument("--n-layers", type=int, default=NUM_ANSATZ_LAYERS)
    p.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=sorted(NOISE_PRESETS),
        choices=sorted(NOISE_PRESETS),
    )
    p.add_argument(
        "--metric-name",
        type=str,
        default="noise_eval_val_acc",
        help="Metric label persisted alongside each benchmark row.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = str(get_device())

    console.rule("[bold cyan]MedQCNN Noise Sensitivity Eval[/bold cyan]")
    console.print(f"  Checkpoint: {args.checkpoint}")
    console.print(f"  Dataset:    {args.dataset}")
    console.print(f"  Presets:    {args.presets}")
    console.print(f"  Device:     {device}")

    _, val_loader, _ = get_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        download=True,
    )
    from medmnist import INFO

    n_classes = len(INFO[args.dataset]["label"])

    # Load weights once. We'll only swap the quantum_layer noise model.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    results: list[tuple[str, float, float, float]] = []
    for preset in args.presets:
        noise_config = get_preset(preset)

        model = HybridQCNN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            n_classes=n_classes,
            pretrained=False,
            noise_config=noise_config if not noise_config.is_noiseless else None,
        )
        # The eval-layer architecture matches the trained one — strict
        # load is safe because we always rebuild with the same n_qubits.
        # Quantum-layer weights load from the source-of-truth Parameter.
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        start = time.time()
        acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start

        console.print(
            f"  {preset:>7s}: depol={noise_config.depolarising_p:.4f} "
            f"readout={noise_config.readout_p:.4f}  acc={acc:.4f}  ({elapsed:.1f}s)"
        )
        results.append(
            (preset, noise_config.depolarising_p, noise_config.readout_p, acc)
        )

    if args.training_run_id is None:
        console.print(
            "\n[bold yellow]--training-run-id not provided; "
            "skipping DB persistence.[/bold yellow]"
        )
        return

    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import create_benchmark

    with db_session() as session:
        for _, depol_p, readout_p, acc in results:
            create_benchmark(
                session,
                training_run_id=args.training_run_id,
                metric_name=args.metric_name,
                metric_value=acc,
                depolarising_p=depol_p,
                readout_p=readout_p,
            )
    console.print(
        f"\n[bold green]Persisted {len(results)} benchmarks "
        f"to training_run_id={args.training_run_id}.[/bold green]"
    )


if __name__ == "__main__":
    main()
