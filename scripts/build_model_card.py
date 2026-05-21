"""
Generate a Model Card for a trained MedQCNN checkpoint.

Output follows the spirit of:
  * Mitchell et al., 2019 (Model Cards for Model Reporting)
  * FDA / Health Canada / MHRA — Good Machine Learning Practice for
    Medical Device Development: Guiding Principles (Jan 2025 update)

The card is rendered as Markdown to ``docs/model_card.md`` (overridable
with ``--output``). It pulls numbers from the checkpoint's
``training_runs`` and ``benchmarks`` rows if present and from the
sidecar JSON files produced by ``scripts/calibrate.py``.

Usage:
    uv run python scripts/build_model_card.py \\
        --checkpoint checkpoints/model_best.pt \\
        --dataset breastmnist
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from medqcnn.config.constants import (  # noqa: E402
    BACKBONE_NAME,
    IMAGE_SIZE,
    NUM_ANSATZ_LAYERS,
    NUM_QUBITS,
)
from medqcnn.utils.logging import console  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a MedQCNN model card.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        default="breastmnist",
        help="Dataset the checkpoint was trained on.",
    )
    p.add_argument("--n-qubits", type=int, default=NUM_QUBITS)
    p.add_argument("--n-layers", type=int, default=NUM_ANSATZ_LAYERS)
    p.add_argument("--output", type=str, default="docs/model_card.md")
    p.add_argument(
        "--intended-use",
        type=str,
        default=(
            "Research-grade decision support for binary / multi-class "
            "medical-image classification on MedMNIST-scale datasets. "
            "Not authorised for clinical use."
        ),
    )
    return p.parse_args()


def read_calibration_sidecar(ckpt_path: Path) -> dict | None:
    side = ckpt_path.with_suffix(ckpt_path.suffix + ".calibration.json")
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except (ValueError, OSError):
        return None


def read_conformal_sidecar(ckpt_path: Path) -> dict | None:
    side = ckpt_path.with_suffix(ckpt_path.suffix + ".conformal.json")
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except (ValueError, OSError):
        return None


def read_ood_sidecar(ckpt_path: Path) -> dict | None:
    side = ckpt_path.with_suffix(ckpt_path.suffix + ".ood.json")
    if not side.exists():
        return None
    try:
        payload = json.loads(side.read_text())
    except (ValueError, OSError):
        return None
    # Strip the (large) mean / covariance arrays; keep only summary fields.
    return {
        k: v
        for k, v in payload.items()
        if k in {"feature_dim", "threshold", "percentile", "n_train", "regularization"}
    }


def latest_training_run(dataset: str) -> dict | None:
    """Best-effort: pull the most recent matching training_run from the DB."""
    try:
        from medqcnn.db.connection import db_session
        from medqcnn.db.crud import list_training_runs

        with db_session() as session:
            rows, _ = list_training_runs(session, offset=0, limit=50)
        for row in rows:
            if row.dataset == dataset:
                return row.to_dict() if hasattr(row, "to_dict") else None
    except Exception:  # noqa: BLE001 — DB may not be configured
        return None
    return None


def render_card(args, ckpt_path: Path) -> str:
    calib = read_calibration_sidecar(ckpt_path)
    conf = read_conformal_sidecar(ckpt_path)
    ood = read_ood_sidecar(ckpt_path)
    run = latest_training_run(args.dataset)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    size_mb = ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else 0.0

    lines: list[str] = []
    add = lines.append

    add(f"# Model Card — MedQCNN ({args.dataset})")
    add("")
    add(f"*Generated {now} for `{ckpt_path}` ({size_mb:.1f} MB).*")
    add("")
    add("## 1. Model details")
    add("")
    add(f"- **Architecture**: Hybrid quantum-classical CNN (`HybridQCNN`).")
    add(
        f"- **Classical backbone**: `{BACKBONE_NAME}` (ImageNet-pretrained, "
        "frozen during MedQCNN training)."
    )
    add(
        f"- **Quantum layer**: {args.n_qubits}-qubit variational circuit, "
        f"{args.n_layers}-layer hardware-efficient ansatz, "
        "amplitude encoding by default. Optional data re-uploading."
    )
    add(
        f"- **Input**: single-channel medical image, resized to "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}."
    )
    add("- **Output**: class logits + softmax probabilities + per-qubit ⟨σ_z⟩.")
    add(
        "- **Trust artefacts**: optional sidecar JSONs (`*.calibration.json`, "
        "`*.conformal.json`, `*.ood.json`) loaded automatically by the API."
    )
    add("")

    add("## 2. Intended use")
    add("")
    add(args.intended_use)
    add("")
    add("**Out of scope**:")
    add("- Any standalone diagnosis without clinician review.")
    add("- Patient cohorts not represented in the training data.")
    add("- Real-time or life-critical decision making.")
    add("")

    add("## 3. Training data")
    add("")
    add(f"- **Dataset**: `{args.dataset}` (MedMNIST family unless overridden).")
    if run:
        add(f"- **Training epochs**: {run.get('epochs', 'n/a')}")
        add(f"- **Batch size**: {run.get('batch_size', 'n/a')}")
        add(f"- **Learning rate**: {run.get('learning_rate', 'n/a')}")
        if run.get("noise_config_json"):
            add(f"- **Noise model**: `{run['noise_config_json']}`")
    else:
        add("- *(No matching training-run row in DB; numbers omitted.)*")
    add("")

    add("## 4. Evaluation")
    add("")
    if run:
        add(
            f"- **Final train accuracy**: {run.get('final_train_acc', 'n/a')}"
        )
        add(
            f"- **Final validation accuracy**: {run.get('final_val_acc', 'n/a')}"
        )
        add(
            f"- **Wall-clock duration**: {run.get('duration_seconds', 'n/a')}s"
        )
    else:
        add("- *(No matching training-run row in DB; numbers omitted.)*")
    add("")
    if calib:
        add("### Calibration")
        add("")
        add(f"- Temperature scaling: `T = {calib.get('temperature', 'n/a'):.4f}`")
        add(f"- ECE before / after: `{calib.get('ece_before', 'n/a'):.4f}` / "
            f"`{calib.get('ece_after', 'n/a'):.4f}`")
        add(f"- Calibration split size: `n = {calib.get('n_calibration', 'n/a')}`")
        add("")
    if conf:
        add("### Conformal prediction (APS, Romano et al. 2020)")
        add("")
        add(f"- Coverage target: `1 - α = {1 - conf.get('alpha', 0.1):.2f}`")
        add(f"- Calibrated quantile `q̂`: `{conf.get('qhat', 'n/a')}`")
        add(f"- Calibration split size: `n = {conf.get('n_calibration', 'n/a')}`")
        add("- Non-singleton prediction sets surface as `abstained=true` "
            "in the `/predict` response.")
        add("")
    if ood:
        add("### Out-of-distribution gate")
        add("")
        add(f"- Method: Gaussian density on `{BACKBONE_NAME}` backbone features")
        add(f"- Feature dimension: `{ood.get('feature_dim', 'n/a')}`")
        add(f"- Threshold percentile on val: `{ood.get('percentile', 'n/a')}`")
        add(f"- Negative log-likelihood threshold: `{ood.get('threshold', 'n/a')}`")
        add("")

    add("## 5. Fairness, ethics, and known limitations")
    add("")
    add(
        "- MedMNIST is a downscaled benchmark (28×28 → upsampled to "
        f"{IMAGE_SIZE}×{IMAGE_SIZE}). Performance on full-resolution "
        "clinical images is not guaranteed."
    )
    add(
        "- Subgroup (skin tone, age band, scanner manufacturer) gaps were "
        "**not** quantified for this checkpoint. Fairness harness "
        "(`medqcnn/eval/fairness.py`) is planned but not yet wired."
    )
    add(
        "- Quantum simulation noise is fixed at training time. Hardware "
        "noise profiles drift; periodic re-calibration is required for any "
        "real-NISQ deployment."
    )
    add("")

    add("## 6. Caveats, recommendations, and operational guidance")
    add("")
    add(
        "- Run `scripts/calibrate.py` after every retrain so the sidecars "
        "stay in sync with the checkpoint."
    )
    add(
        "- Run `scripts/multi_seed_sweep.py --aggregate` to publish CIs "
        "across at least 5 seeds before any external evaluation claim."
    )
    add(
        "- Treat any prediction with `abstained=true` as requiring "
        "human review."
    )
    add(
        "- Treat any prediction whose backbone features score above the OOD "
        "threshold as out-of-distribution and refuse to act."
    )
    add("")

    add("## 7. Provenance")
    add("")
    add(f"- Checkpoint path: `{ckpt_path}`")
    add(f"- Card generator: `scripts/build_model_card.py`")
    add(
        "- Backed by the GMLP guiding principles "
        "(FDA / Health Canada / MHRA, Jan 2025)."
    )
    add("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    card = render_card(args, ckpt_path)
    out_path.write_text(card)
    console.print(f"[bold green]Wrote model card to {out_path}[/bold green]")
    console.print(f"  {len(card.splitlines())} lines, {len(card)} bytes")

    # Suppress unused-import lint if a path doesn't touch torch.
    _ = torch


if __name__ == "__main__":
    main()
