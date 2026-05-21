"""
Post-training calibration + conformal-fit script.

Loads a HybridQCNN checkpoint, collects logits on the validation
loader, fits temperature scaling, then fits a split-conformal APS
predictor on the temperature-calibrated probabilities. Two sidecar
JSONs are written next to the checkpoint:

  * ``<ckpt>.calibration.json``
  * ``<ckpt>.conformal.json``

``ModelService`` picks them up automatically when the checkpoint is
served, no API changes required.

Usage:
    uv run python scripts/calibrate.py \\
        --checkpoint checkpoints/model_best.pt \\
        --dataset breastmnist \\
        --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from medqcnn.data.loader import get_medmnist_loaders
from medqcnn.inference.conformal import ConformalPredictor
from medqcnn.inference.ood import OODDetector
from medqcnn.model.hybrid import HybridQCNN
from medqcnn.training.calibration import (
    apply_temperature,
    compute_ece,
    fit_temperature,
)
from medqcnn.utils.device import get_device, set_seed
from medqcnn.utils.logging import console


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate + fit conformal predictor")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="breastmnist")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal miscoverage rate (coverage target 1-alpha).",
    )
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-ood",
        action="store_true",
        help="Skip fitting the Gaussian OOD detector on backbone features.",
    )
    p.add_argument(
        "--ood-percentile",
        type=float,
        default=95.0,
        help="In-distribution percentile used to set the OOD NLL threshold.",
    )
    return p.parse_args()


def collect_logits(
    model: torch.nn.Module, loader, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (logits, labels, backbone_features).

    backbone_features is None when the model has no ``backbone`` attribute.
    """
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    has_backbone = hasattr(model, "backbone")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            if images.shape[-1] != 224:
                images = torch.nn.functional.interpolate(
                    images, size=(224, 224), mode="bilinear", align_corners=False
                )
            labels = labels.to(device).squeeze().long()
            logits = model(images)
            logits_chunks.append(logits.cpu())
            label_chunks.append(labels.cpu())
            if has_backbone:
                feature_chunks.append(model.backbone(images).cpu())
    features = torch.cat(feature_chunks) if feature_chunks else None
    return torch.cat(logits_chunks), torch.cat(label_chunks), features


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        console.print(f"[bold red]Checkpoint not found: {ckpt_path}[/bold red]")
        sys.exit(1)

    set_seed(args.seed)
    device = get_device()

    console.rule("[bold cyan]Loading checkpoint[/bold cyan]")
    checkpoint = torch.load(ckpt_path, map_location=device)
    labels = checkpoint.get("labels") or ["class_0", "class_1"]
    n_classes = len(labels)

    model = HybridQCNN(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=n_classes,
        pretrained=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    console.rule("[bold cyan]Collecting train + validation logits[/bold cyan]")
    train_loader, val_loader, _ = get_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        download=True,
    )
    logits, val_labels, val_features = collect_logits(model, val_loader, device)
    console.print(f"  Collected {logits.shape[0]} validation samples")
    train_features: torch.Tensor | None = None
    if not args.skip_ood:
        _, _, train_features = collect_logits(model, train_loader, device)
        if train_features is not None:
            console.print(f"  Collected {train_features.shape[0]} train features")

    probs_before = torch.softmax(logits, dim=1)
    ece_before = compute_ece(probs_before, val_labels)
    console.print(f"  ECE (pre-calibration):  {ece_before:.4f}")

    console.rule("[bold cyan]Fitting temperature[/bold cyan]")
    temperature = fit_temperature(logits, val_labels)
    probs_after = torch.softmax(apply_temperature(logits, temperature), dim=1)
    ece_after = compute_ece(probs_after, val_labels)
    console.print(f"  Fitted T = {temperature:.4f}")
    console.print(f"  ECE (post-calibration): {ece_after:.4f}")

    calib_path = ckpt_path.with_suffix(ckpt_path.suffix + ".calibration.json")
    calib_path.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "n_calibration": int(logits.shape[0]),
            },
            indent=2,
        )
    )
    console.print(f"  Wrote {calib_path}")

    console.rule("[bold cyan]Fitting conformal predictor[/bold cyan]")
    conformal = ConformalPredictor(alpha=args.alpha)
    qhat = conformal.calibrate(probs_after, val_labels)
    sets = conformal.predict_set(probs_after)
    coverage = sum(int(int(val_labels[i].item()) in s) for i, s in enumerate(sets))
    coverage_rate = coverage / max(1, len(sets))
    avg_size = sum(len(s) for s in sets) / max(1, len(sets))
    console.print(f"  alpha = {args.alpha}, qhat = {qhat:.4f}")
    console.print(
        f"  empirical coverage = {coverage_rate:.4f} (target ≥ {1 - args.alpha:.2f})"
    )
    console.print(f"  avg set size = {avg_size:.3f}")

    conf_path = ckpt_path.with_suffix(ckpt_path.suffix + ".conformal.json")
    conformal.save(conf_path)
    console.print(f"  Wrote {conf_path}")

    if not args.skip_ood and train_features is not None and val_features is not None:
        console.rule("[bold cyan]Fitting OOD Gaussian on backbone features[/bold cyan]")
        ood = OODDetector()
        ood.fit(train_features)
        threshold = ood.calibrate_threshold(
            val_features, percentile=args.ood_percentile
        )
        console.print(
            f"  Threshold (P{args.ood_percentile:.0f} of in-dist val NLL): "
            f"{threshold:.4f}"
        )
        console.print(f"  Feature dim = {ood.feature_dim}, n_train = {ood.n_train}")
        ood_path = ckpt_path.with_suffix(ckpt_path.suffix + ".ood.json")
        ood.save(ood_path)
        console.print(f"  Wrote {ood_path}")


if __name__ == "__main__":
    main()
