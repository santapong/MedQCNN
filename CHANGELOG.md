# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — Docker & CI/CD Improvements

### Added
- **`.dockerignore`** — Excludes `.venv/` (7.9 GB), `node_modules/`, `.git/`, `data/`, `checkpoints/`, docs, and test caches from Docker build context. Drastically reduces image build time and size.
- **GitHub Actions Docker workflow** (`.github/workflows/docker.yml`) — Builds Docker image on every push/PR, pushes to DockerHub on main branch and version tags with multi-tag strategy (latest, sha, semver).
- **`.env.example`** — Documents all environment variables: `DATABASE_URL`, `API_HOST`, `API_PORT`, `CHECKPOINT_PATH`, `N_QUBITS`, Kafka settings, and Docker Compose overrides.

### Changed
- **Dockerfile** — Replaced `COPY checkpoints/` (fails when dir is empty/excluded) with `RUN mkdir -p checkpoints data logs` for volume mount targets. Added `curl` for health check. Uses `curl -f` instead of Python one-liner for `HEALTHCHECK`. Sets `DATABASE_URL` default to SQLite fallback.

## [0.2.0] — Phase 5: Interactive Frontend & Database

### Added — Database Layer (Sprint 5.1)
- **SQLAlchemy ORM** — `medqcnn/db/` package with `models.py` (Prediction, TrainingRun, Benchmark tables), `connection.py` (engine/session management), and `crud.py` (create, list, get operations).
- **PostgreSQL support** — Production database via Docker Compose. Falls back to SQLite for local development.
- **Auto-storage** — Every `/predict` and `/predict/batch` result is automatically persisted to the database.
- **New API endpoints**:
  - `GET /predictions` — Paginated prediction history with label, confidence, and filename filters
  - `GET /predictions/{id}` — Single prediction detail with full quantum analysis
  - `GET /training-runs` — List training runs with metrics and loss/accuracy history
  - `GET /training-runs/{id}` — Single training run detail
  - `GET /benchmarks` — Aggregated benchmark metrics, filterable by training run

### Added — Batch Prediction (Sprint 5.2)
- **`POST /predict/batch`** — Accepts up to 100 base64-encoded images, returns per-image results with summary statistics (benign/malignant count, average confidence).
- **Upload limit** increased from 10 MB to 250 MB per request.
- **Frontend batch page** (`/batch`) — Multi-file drag-and-drop upload, file list with individual remove, batch results table with expandable rows showing probabilities and quantum values.

### Added — Training & Benchmark Dashboards (Sprint 5.3)
- **Trainer auto-storage** — `Trainer.fit()` now persists training run metadata (dataset, hyperparams, final metrics, full loss/accuracy history) to the database on completion. Key metrics are also stored as benchmark entries.
- **Evaluate auto-storage** — `scripts/evaluate.py` stores test accuracy, F1, and AUC-ROC in the database.
- **Training page** (`/training`) — Runs table with metrics, interactive Recharts loss/accuracy curve charts for selected run.
- **Benchmarks page** (`/benchmarks`) — Reference comparison cards (4-qubit vs 8-qubit vs classical baseline), horizontal bar charts for parameter count, memory usage, and inference latency.
- **Recharts** added as frontend dependency for interactive data visualization.

### Added — Prediction History & UX Polish (Sprint 5.4)
- **History page** (`/history`) — Paginated prediction list with label filter, filename search, and CSV export.
- **Detail page** (`/history/{id}`) — Full prediction detail with classification, probabilities, quantum expectation values, and metadata.
- **Dark/light theme toggle** — ThemeToggle component with localStorage persistence and CSS variable-based theming.
- **Loading skeletons** — Skeleton, CardSkeleton, and TableSkeleton components for all data-fetching pages.
- **Updated Navbar** — New navigation links for Batch, History, Training, Benchmarks pages.

### Changed
- **Docker Compose** — Added PostgreSQL 16 service with health check, persistent volume, and `DATABASE_URL` env var.
- **Version** bumped to 0.2.0 across `pyproject.toml`, `package.json`, and API schemas.
- **Dependencies** — Added `sqlalchemy>=2.0.0` and `psycopg2-binary>=2.9.0` to Python deps; `recharts` to frontend deps.

## [0.1.0] — Phases 0–4

### Added — Next.js Web Dashboard (Frontend)
- **Next.js 16 + Bun** — Full web dashboard in `frontend/` built with Next.js, TypeScript, and Tailwind CSS v4, managed with Bun.
- **Diagnose Page** (`/`) — Drag-and-drop image upload with live preview, automatic quantum inference via the REST API, and results display including:
  - Benign/Malignant classification with confidence score
  - Probability bar visualization
  - Per-qubit Pauli-Z expectation value chart with color-coded bars
  - Interactive pipeline step indicator (Image → ResNet-18 → Projector → Quantum → Classify)
  - Medical disclaimer
- **Model Info Page** (`/model`) — Live model architecture dashboard showing:
  - Service health status from `/health` endpoint
  - Quantum configuration (qubits, latent dim, ansatz layers)
  - Trainable parameter breakdown by component
  - Visual pipeline architecture diagram
- **Components** — `Navbar`, `ImageUploader` (drag-and-drop), `DiagnosisResult`, `QuantumBar` (expectation value visualizer)
- **API Client** (`src/lib/api.ts`) — Typed fetch wrappers for `/health`, `/info`, `/predict` with base64 encoding utility
- **API Proxy** — Next.js rewrites proxy `/api/*` to `localhost:8000` for seamless backend integration

### Added — Project Completion & CI/CD
- **Data Module** — `medqcnn/data/` with `loader.py` (MedMNIST DataLoaders) and `preprocessing.py` (normalize, resize, slice extraction, pipeline)
- **GitHub Actions CI** — `.github/workflows/ci.yml` with lint (ruff) and test (pytest) jobs
- **API Input Validation** — Size limits, base64 validation, format checks, dimension limits on `/predict`
- **Benchmarks** — `docs/benchmarks.md` with parameter counts, memory profile, latency measurements

### Fixed
- `pyproject.toml`: `dotenv` → `python-dotenv` (correct PyPI package), added missing `langgraph` dependency
- `.gitignore`: Fixed `data/` pattern to only match top-level directory, not `medqcnn/data/`
- `test_model.py`: Use `pretrained=False` to avoid network dependency in tests
- Removed unused imports and fixed ruff lint warnings across all Python sources

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
