# MedQCNN — Hybrid Quantum-Classical CNN for Medical Diagnostics

A hybrid quantum-classical neural network for medical image classification, designed for edge deployment on Raspberry Pi 5 clusters.

## Architecture

```
Medical Image → [ResNet-18 (frozen)] → FC Projector → L2 Norm
    → [Amplitude Encoding] → [Variational Ansatz (Ry/Rz/CZ)]
    → [Pauli-Z Measurement] → Classifier → Diagnosis
```

| Component | Role | Module |
|-----------|------|--------|
| **Node A** — Classical | Feature extraction + compression to R^256 | `medqcnn/classical/` |
| **Node B** — Quantum | Amplitude encoding + HEA + local Pauli-Z | `medqcnn/quantum/` |
| **Hybrid Model** | End-to-end differentiable pipeline | `medqcnn/model/` |
| **API Server** | REST endpoints for inference + history | `medqcnn/api/` |
| **Database** | Prediction history, training runs, benchmarks | `medqcnn/db/` |
| **MCP Server** | AI agent tool integration | `medqcnn/mcp/` |

## Quick Start

```bash
# Install dependencies
uv sync

# Verify environment
uv run python main.py

# Run end-to-end demo (4-qubit, BreastMNIST)
uv run python scripts/demo.py

# Train the model
uv run python scripts/train.py --epochs 10 --n-qubits 4

# Evaluate
uv run python scripts/evaluate.py --n-qubits 4

# Start REST API
uv run python scripts/serve.py

# Start MCP server (for AI agents)
uv run python scripts/mcp_server.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/info` | Model architecture details |
| `POST` | `/predict` | Inference (base64 image -> diagnosis) |
| `POST` | `/predict/batch` | Batch inference (multiple images) |
| `GET` | `/predictions` | Prediction history (paginated, filterable) |
| `GET` | `/predictions/{id}` | Single prediction detail |
| `GET` | `/training-runs` | List training runs with metrics |
| `GET` | `/training-runs/{id}` | Single training run detail |
| `GET` | `/benchmarks` | Aggregated benchmark data |

Example:
```bash
# Health check
curl http://localhost:8000/health

# Single predict (base64 image)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-encoded-image>"}'

# Batch predict
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"images": ["<base64-img-1>", "<base64-img-2>"]}'

# Prediction history
curl http://localhost:8000/predictions?limit=20&label=Benign

# Training runs
curl http://localhost:8000/training-runs
```

## MCP Tools (AI Agent Integration)

| Tool | Description |
|------|-------------|
| `diagnose` | Run quantum inference on a medical image |
| `model_info` | Get model architecture and parameter counts |
| `list_datasets` | List available MedMNIST datasets |

Configure in your MCP client:
```json
{
  "mcpServers": {
    "medqcnn": {
      "command": "uv",
      "args": ["run", "python", "scripts/mcp_server.py"],
      "cwd": "/path/to/MedQCNN"
    }
  }
}
```

## Web Dashboard (Frontend)

The project includes a Next.js web dashboard for interactive diagnosis, batch processing, and analytics.

```bash
# Start the backend API
uv run python scripts/serve.py

# In a separate terminal, start the frontend
cd frontend
bun install
bun run dev
# Open http://localhost:3000
```

**Pages:**
- `/` — Upload medical images for quantum-classical diagnosis with real-time results
- `/batch` — Batch upload multiple images with summary statistics and expandable results
- `/history` — Browse, filter, and export prediction history with CSV download
- `/history/{id}` — Detailed prediction view with quantum analysis breakdown
- `/training` — Training dashboard with runs table and interactive loss/accuracy charts
- `/benchmarks` — Parameter count, memory usage, and inference latency comparisons
- `/model` — View model architecture, parameter counts, and service health

**Features:**
- Dark/light theme toggle
- Loading skeletons for all data-fetching pages
- Responsive mobile layout
- Interactive Recharts visualizations

```bash
# Production build
cd frontend && bun run build && bun run start
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

Key variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///medqcnn.db` | PostgreSQL or SQLite connection string |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `CHECKPOINT_PATH` | — | Path to trained model checkpoint |
| `N_QUBITS` | `4` | Number of qubits (4 = demo, 8 = production) |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |

## Docker Deployment

```bash
# Build and run with PostgreSQL + Kafka
docker compose up -d

# API available at http://localhost:8000
# PostgreSQL at localhost:5432
# Kafka broker at localhost:9092
```

The `.dockerignore` excludes `.venv/`, `node_modules/`, `.git/`, `data/`, and `checkpoints/` from the build context. Data and checkpoints are mounted as Docker volumes instead.

**CI/CD:** The GitHub Actions workflow (`.github/workflows/docker.yml`) automatically builds the Docker image on every push and pushes to DockerHub on main branch. Set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` as GitHub repository secrets.

## Database

MedQCNN uses SQLAlchemy with PostgreSQL (production) or SQLite (development fallback).

**Tables:**
- `predictions` — Every inference result with probabilities, quantum values, and metadata
- `training_runs` — Training session metadata with history (loss/accuracy curves)
- `benchmarks` — Per-metric values linked to training runs

Set `DATABASE_URL` environment variable for PostgreSQL:
```bash
export DATABASE_URL=postgresql://medqcnn:medqcnn@localhost:5432/medqcnn
```

Without `DATABASE_URL`, falls back to SQLite (`medqcnn.db` in project root).

## Project Structure

```
MedQCNN/
├── medqcnn/                 # Core package
│   ├── config/              # Constants, settings
│   ├── data/                # Data loaders, preprocessing
│   ├── classical/           # ResNet backbone, FC projector
│   ├── quantum/             # Encoding, ansatz, observables, QNode
│   ├── model/               # HybridQCNN nn.Module
│   ├── training/            # Trainer, loss, metrics, visualization
│   ├── api/                 # Litestar REST server, Kafka handler
│   ├── db/                  # SQLAlchemy models, connection, CRUD
│   ├── mcp/                 # MCP server for AI agents
│   ├── agent/               # LangChain tools, LangGraph agent
│   └── utils/               # Logging, device management
├── scripts/                 # CLI scripts (demo, train, serve, mcp)
├── frontend/                # Next.js web dashboard (Bun)
├── tests/                   # Unit tests
├── notebooks/               # Educational notebooks
├── Dockerfile               # Multi-stage container
├── docker-compose.yml       # API + PostgreSQL + Kafka orchestration
└── GEMINI.md                # Full architecture spec
```

## Hardware Constraints

- **Qubits:** 8 max (256-dim latent space)
- **Ansatz layers:** 4 (shallow — barren plateau mitigation)
- **Target:** CPU-only, 16-32 GB RAM, Raspberry Pi 5 cluster
- **Simulation:** State-vector via PennyLane `default.qubit`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Classical Vision | PyTorch, torchvision (ResNet-18) |
| Quantum Circuit | PennyLane (TorchLayer, backprop) |
| Database | PostgreSQL 16 / SQLite (SQLAlchemy 2.x) |
| API Server | Litestar, Uvicorn |
| Agent Protocol | MCP (Model Context Protocol) |
| Message Broker | Apache Kafka |
| Containerization | Docker, Docker Compose |
| Frontend | Next.js 16, TypeScript, Tailwind CSS v4, Recharts |
| Frontend Runtime | Bun |
| Package Manager | uv (Python), Bun (JS) |

## Testing

```bash
# Install dev deps
uv sync --extra dev

# Run tests
uv run python -m pytest tests/ -v

# Lint
uv run ruff check .
```

## License

MIT
