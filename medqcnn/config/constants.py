"""
Project-wide constants for MedQCNN.

These values are derived from the architectural constraints defined
in GEMINI.md — specifically the 8-qubit cap for edge-compute
deployment on Raspberry Pi 5 cluster hardware.
"""

# ──────────────────────────────────────────────
# Quantum Architecture Constraints
# ──────────────────────────────────────────────
NUM_QUBITS: int = 8
"""Number of qubits — capped at 8 to prevent memory swapping on
16–32 GB edge devices. This is a hard constraint."""

LATENT_DIM: int = 2**NUM_QUBITS  # 256
"""Dimensionality of the classical latent vector z ∈ R^256.
Must equal 2^NUM_QUBITS for amplitude encoding compatibility."""

NUM_ANSATZ_LAYERS: int = 4
"""Number of variational ansatz layers. Kept shallow to mitigate
barren plateaus (Var(∂L/∂θ) ∝ 2^{-n})."""

# ──────────────────────────────────────────────
# Classical Vision Settings
# ──────────────────────────────────────────────
IMAGE_SIZE: int = 224
"""Input image spatial resolution (H=W) for the ResNet backbone."""

BACKBONE_NAME: str = "resnet18"
"""Pre-trained backbone architecture. ResNet-18 chosen for
lightweight edge deployment."""

# ──────────────────────────────────────────────
# Training Defaults
# ──────────────────────────────────────────────
SEED: int = 42
"""Global random seed for reproducibility across PyTorch,
NumPy, and PennyLane."""

DEFAULT_LEARNING_RATE: float = 1e-3
"""Default Adam optimizer learning rate."""

DEFAULT_BATCH_SIZE: int = 16
"""Default batch size — conservative for edge RAM constraints."""

DEFAULT_EPOCHS: int = 50
"""Default number of training epochs."""

# ──────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────
DATA_DIR: str = "data"
CHECKPOINT_DIR: str = "checkpoints"
LOG_DIR: str = "logs"
