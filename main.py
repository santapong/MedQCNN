"""
MedQCNN — CLI Entrypoint

Main entry point for the hybrid quantum-classical medical
diagnostics pipeline. Run with:
    uv run python main.py
"""

from __future__ import annotations

from medqcnn.utils.device import get_device, get_memory_info, set_seed
from medqcnn.utils.logging import console, setup_logger

logger = setup_logger()


def main() -> None:
    """MedQCNN main entrypoint."""
    from medqcnn import __version__
    from medqcnn.config.constants import (
        LATENT_DIM,
        NUM_ANSATZ_LAYERS,
        NUM_QUBITS,
    )

    console.rule("[bold cyan]MedQCNN: Hybrid Quantum-Classical CNN[/bold cyan]")
    console.print(f"  Version:       {__version__}")
    console.print(f"  Qubits:        {NUM_QUBITS}")
    console.print(f"  Latent dim:    {LATENT_DIM}")
    console.print(f"  Ansatz layers: {NUM_ANSATZ_LAYERS}")

    # Set reproducibility
    set_seed()

    # Device info
    device = get_device()
    console.print(f"  Device:        {device}")

    mem = get_memory_info()
    console.print(
        f"  RAM:           {mem['ram_used_gb']:.1f} / "
        f"{mem['ram_total_gb']:.1f} GB ({mem['ram_percent']}%)"
    )

    console.rule("[bold green]System Ready[/bold green]")
    logger.info("MedQCNN initialized successfully.")


if __name__ == "__main__":
    main()
