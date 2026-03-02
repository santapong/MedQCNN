# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — LangChain Agent Integration (CaaS-Q)
- **LangChain Tools** — `medqcnn/agent/tools.py` with 3 `@tool`-decorated functions:
  - `quantum_diagnose` — run quantum-classical inference on medical images
  - `get_model_info` — model architecture and parameter details
  - `list_medical_datasets` — available MedMNIST benchmark datasets
- **Medical Diagnostic Agent** — `medqcnn/agent/agent.py` with:
  - `create_agent_executor()` — builds a LangGraph ReAct agent with MedQCNN tools
  - `run_diagnostic_without_llm()` — standalone clinical report generator (no API key needed)
  - System prompt for clinical report generation with disclaimers
- **CaaS-Q Use Case Notebook** — `notebooks/02_caasq_agent_usecase.ipynb` demonstrating:
  - MedQCNN as a LangChain tool
  - Full diagnostic flow with quantum expectation value visualization
  - Clinical report generation
  - Integration options (LangChain, MCP, REST, Kafka)
- **Dependencies** — `langchain`, `langchain-core`, `langgraph` added.

### Added — Sprint 4: CaaS-Q Deployment & MCP
- **Litestar REST API** — `medqcnn/api/server.py` with `GET /health`, `GET /info`, `POST /predict` endpoints for medical image inference.
- **MCP Server** — `medqcnn/mcp/server.py` exposing 3 tools for AI agent integration:
  - `diagnose` — quantum-classical inference on medical images
  - `model_info` — model architecture and parameter details
  - `list_datasets` — available MedMNIST benchmark datasets
- **Kafka Integration** — `medqcnn/api/kafka_handler.py` with async producer/consumer for event-driven inference pipeline.
- **Docker** — Multi-stage `Dockerfile` (deps + runtime) and `docker-compose.yml` with KRaft-mode Kafka.
- **Scripts** — `scripts/serve.py` (API server), `scripts/mcp_server.py` (MCP server).
- **README** — Complete project documentation with architecture, API docs, MCP config, Docker usage.
- **Dependencies** — `litestar`, `uvicorn`, `mcp` added.

### Added — Sprint 3: Training & Benchmarking
- **Training Visualization** — `training/visualization.py` with `plot_training_history()`, `plot_roc_curve()`, `plot_confusion_matrix()` for publication-quality training metrics plots.
- **Evaluation Script** — `scripts/evaluate.py` for test set evaluation with accuracy, AUC-ROC, F1, and auto-generated visualization plots.
- **Educational Notebook** — `notebooks/01_medqcnn_explained.ipynb` covering:
  - Problem motivation (parameter bloat in medical imaging)
  - Classical CNN compression pipeline walkthrough
  - Quantum circuit math (amplitude encoding, HEA ansatz, Pauli-Z measurements)
  - Barren plateau problem and mitigation strategies
  - Live model demo and training on BreastMNIST with loss/accuracy curves
- **Jupyter Dependencies** — Added `jupyter` and `ipykernel` for notebook support.

### Added — Sprint 2: The Quantum Bridge
- **Differentiable Quantum Pipeline** — Integrated PennyLane `qml.qnn.TorchLayer` into `HybridQCNN`, enabling native PyTorch autograd gradient flow through the quantum circuit. Replaces the `.detach().numpy()` approach that broke backpropagation.
- **Per-Qubit Measurements** — Quantum layer now outputs `n_qubits` individual ⟨σ_z⟩ expectation values instead of a single averaged scalar, giving the classifier richer features.
- **`create_torch_qnode()`** — New factory in `quantum/qnode.py` producing TorchLayer-compatible circuits with `interface='torch'` and `diff_method='backprop'`.
- **`create_quantum_layer()`** — Convenience function returning a ready-to-use `nn.Module` wrapping the quantum circuit.
- **Projector Refinements** — Added `BatchNorm1d` layers and Kaiming uniform weight initialization to `classical/projector.py` for training stability.
- **`DEMO_QUBITS = 4`** — New constant for fast 4-qubit demos (16-dim latent space).
- **`scripts/demo.py`** — End-to-end demo: loads BreastMNIST, runs HybridQCNN forward pass, verifies gradient flow.
- **`scripts/train.py`** — CLI training script with argparse (`--dataset`, `--epochs`, `--batch-size`, `--lr`, `--n-qubits`).
- **Starter Workflow** — `_agents/workflows/run.md` documenting all run commands (setup, demo, test, train, lint).
- **Dev Dependencies** — Installed `pytest`, `pytest-cov`, `ruff`, `mypy` into project venv via `uv sync --extra dev`.

### Added — Sprint 1: The Foundation
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
