"""
Multi-seed × multi-noise grid runner for the S4 stability harness.

Loops over the cartesian product of ``--seeds`` and ``--noise-presets``,
invoking ``scripts/train.py`` once per cell as a subprocess. Each cell's
training run is auto-persisted by :class:`medqcnn.training.trainer.Trainer`
to the ``training_runs`` table, so post-hoc aggregation (mean ± std,
95% CI) reads straight from the DB without bespoke bookkeeping here.

Defaults match the agreed paper-grade grid:
  * 5 seeds: 42 43 44 45 46
  * 4 noise presets: none low medium high
  * total: 20 runs per dataset

Usage:
    uv run python scripts/multi_seed_sweep.py \\
        --dataset breastmnist --epochs 10 --aggregate
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from medqcnn.quantum.noise import NOISE_PRESETS  # noqa: E402
from medqcnn.utils.logging import console  # noqa: E402

DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_PRESETS = ("none", "low", "medium", "high")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed × noise grid runner")
    p.add_argument("--dataset", type=str, default="breastmnist")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated random seeds.",
    )
    p.add_argument(
        "--noise-presets",
        type=str,
        default=",".join(DEFAULT_PRESETS),
        help=f"Comma-separated subset of {sorted(NOISE_PRESETS)}.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default="checkpoints/sweep",
        help="Per-cell checkpoints go into <root>/<noise>/seed_<n>/.",
    )
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save-every", type=int, default=100,
                   help="Disable periodic checkpoints; only best/final are saved.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="After the sweep, print mean ± std and 95%% CI per noise preset "
             "by reading from the training_runs table.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if a single cell fails (default: stop on first error).",
    )
    return p.parse_args()


def parse_int_list(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_preset_list(spec: str) -> list[str]:
    presets = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [p for p in presets if p not in NOISE_PRESETS]
    if bad:
        raise SystemExit(
            f"Unknown noise preset(s) {bad!r}. Valid: {sorted(NOISE_PRESETS)}."
        )
    return presets


def cell_command(args, seed: int, preset: str, ckpt_dir: Path) -> list[str]:
    return [
        sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--n-qubits", str(args.n_qubits),
        "--n-layers", str(args.n_layers),
        "--seed", str(seed),
        "--noise", preset,
        "--save-every", str(args.save_every),
        "--patience", str(args.patience),
    ]


def aggregate(dataset: str, seeds: list[int], presets: list[str]) -> None:
    """Read the most-recent runs matching the sweep config and summarise."""
    from medqcnn.db.connection import db_session
    from medqcnn.db.crud import list_training_runs

    expected = len(seeds) * len(presets)
    with db_session() as session:
        rows, total = list_training_runs(session, offset=0, limit=200)

    cells: dict[str, list[float]] = {p: [] for p in presets}
    for row in rows[:expected * 2]:  # tolerate up to one prior sweep
        if row.dataset != dataset:
            continue
        noise_label = "none"
        if row.noise_config_json:
            import json as _json
            try:
                cfg = _json.loads(row.noise_config_json)
                train_cfg = cfg.get("train") or {}
                if train_cfg.get("depolarising_p", 0.0) >= 0.01:
                    noise_label = "high"
                elif train_cfg.get("depolarising_p", 0.0) >= 0.005:
                    noise_label = "medium"
                elif train_cfg.get("depolarising_p", 0.0) >= 0.001:
                    noise_label = "low"
            except (ValueError, KeyError):
                pass
        if noise_label in cells and row.final_val_acc is not None:
            cells[noise_label].append(float(row.final_val_acc))

    console.rule("[bold green]Sweep summary[/bold green]")
    console.print(f"  Dataset: {dataset}; total rows visible: {total}")
    for preset in presets:
        values = cells[preset][: len(seeds)]
        if not values:
            console.print(f"  {preset:>6s}: (no rows found)")
            continue
        mu = statistics.mean(values)
        sigma = statistics.pstdev(values) if len(values) > 1 else 0.0
        # Normal-approx 95% CI
        half = 1.96 * sigma / max(1.0, (len(values) ** 0.5)) if len(values) > 1 else 0.0
        console.print(
            f"  {preset:>6s}: val_acc = {mu:.4f} ± {sigma:.4f} "
            f"(n={len(values)}, 95% CI ± {half:.4f})"
        )


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    presets = parse_preset_list(args.noise_presets)

    console.rule("[bold cyan]Multi-seed × noise sweep[/bold cyan]")
    console.print(f"  Dataset:   {args.dataset}")
    console.print(f"  Seeds:     {seeds}")
    console.print(f"  Presets:   {presets}")
    console.print(f"  Total cells: {len(seeds) * len(presets)}")
    console.print(f"  Epochs/cell: {args.epochs}")

    ckpt_root = Path(args.checkpoint_root)
    failed: list[tuple[int, str, int]] = []

    for preset in presets:
        for seed in seeds:
            cell_dir = ckpt_root / preset / f"seed_{seed}"
            cmd = cell_command(args, seed, preset, cell_dir)
            console.rule(f"[yellow]cell: noise={preset} seed={seed}[/yellow]")
            console.print(" ".join(cmd))
            if args.dry_run:
                continue
            cell_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            if result.returncode != 0:
                failed.append((seed, preset, result.returncode))
                if not args.continue_on_error:
                    console.print(
                        f"[bold red]Cell failed (rc={result.returncode}); "
                        "stopping. Pass --continue-on-error to override.[/bold red]"
                    )
                    sys.exit(result.returncode)

    if failed:
        console.print(f"[bold red]{len(failed)} cell(s) failed:[/bold red]")
        for seed, preset, rc in failed:
            console.print(f"  noise={preset} seed={seed} rc={rc}")

    if args.aggregate and not args.dry_run:
        aggregate(args.dataset, seeds, presets)


if __name__ == "__main__":
    main()
