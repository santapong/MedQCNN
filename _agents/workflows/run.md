---
description: How to run the MedQCNN project (setup, train, test, demo)
---

# Running MedQCNN

// turbo-all

## 1. Prerequisites

Make sure you have Python 3.11+ and `uv` installed:

```bash
python --version   # should be >= 3.11
uv --version       # should be >= 0.6.0
```

## 2. Install Dependencies

```bash
# Install all runtime dependencies
uv sync

# Install with dev tools (pytest, ruff, mypy)
uv sync --extra dev
```

## 3. Verify the Environment

```bash
# Quick import check
uv run python -c "import medqcnn; print(f'MedQCNN v{medqcnn.__version__}')"

# Run the system info CLI
uv run python main.py
```

This will print your system configuration:
- Qubit count (8), latent dim (256), ansatz layers (4)
- Device (cpu/cuda)
- RAM usage

## 4. Run the Demo (End-to-End Forward Pass)

```bash
# Run the demo script with MedMNIST sample data
uv run python scripts/demo.py
```

This downloads a small MedMNIST dataset, runs one batch through the full hybrid pipeline (ResNet → Projector → Quantum Circuit → Classifier), and prints the output.

## 5. Run Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_quantum.py -v

# Run with coverage
uv run pytest --cov=medqcnn
```

## 6. Lint & Format

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix lint errors
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## 7. Train the Model

```bash
# Train with default settings (BreastMNIST, 50 epochs)
uv run python scripts/train.py

# Train with custom settings
uv run python scripts/train.py --dataset breastmnist --epochs 20 --batch-size 8 --lr 0.001
```

## 8. Project Configuration

All settings are in `settings/settings.json`:
- **Quantum:** qubits, ansatz layers, diff method
- **Classical:** backbone, image size, freeze toggle
- **Training:** learning rate, batch size, epochs, dataset
- **Hardware:** target device, max RAM

Architectural constants are in `medqcnn/config/constants.py`.
