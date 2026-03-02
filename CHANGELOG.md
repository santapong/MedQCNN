# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Project Structure** — Complete `medqcnn/` Python package with 7 sub-modules:
  - `config/` — Project-wide constants (`NUM_QUBITS=8`, `LATENT_DIM=256`, training defaults)
  - `data/` — NIfTI/MedMNIST data loaders, OpenCV preprocessing, augmentation transforms
  - `classical/` — Frozen ResNet-18 backbone + FC projector to ℝ²⁵⁶ with L2 normalization
  - `quantum/` — PennyLane amplitude encoding, HEA variational ansatz (Ry/Rz + CZ), local Pauli-Z observables, QNode with parameter-shift gradients
  - `model/` — `HybridQCNN` end-to-end `nn.Module` (classical → quantum → classifier)
  - `training/` — Training loop with Adam optimizer, hybrid loss with L2 reg, AUC-ROC/F1 metrics
  - `utils/` — Rich-based structured logging, device management, seed reproducibility
- **Test Suite** — Unit tests for data preprocessing, classical backbone, quantum circuit, and hybrid model (`tests/`)
- **CLI Entrypoint** — `main.py` with Rich console output showing system info and config
- **Project Configuration** — `settings/settings.json` with quantum, classical, training, and hardware defaults
- **New Dependencies** — `scikit-learn`, `medmnist`, `matplotlib`, `tqdm`, `rich`, `psutil`
- **Dev Dependencies** — `pytest`, `pytest-cov`, `ruff`, `mypy` (via `[project.optional-dependencies.dev]`)
- **Tool Configs** — Ruff linter/formatter and pytest configs in `pyproject.toml`
- **Scaffolding Directories** — `notebooks/`, `data/`, `checkpoints/`, `docs/` with `.gitkeep`
- **Gitignore** — Medical image formats (`.nii`, `.dcm`), `data/`, `checkpoints/`, `logs/`
- Comprehensive project scope, architecture, and roadmap to `GEMINI.md`.
- Initial project setup with `uv` package manager.
- Core dependencies configured (PyTorch, PennyLane, Qiskit, NumPy, Pandas, NiBabel, OpenCV).
- `GEMINI.md` for Gemini CLI workspace configuration.
- Basic `README.md` and `CHANGELOG.md` structures.
