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

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](./docs/architecture.md) | System design, data flow, integration points |
| [API Reference](./docs/api_reference.md) | REST endpoint specifications |
| [KeepSave Integration](./docs/keepsave_integration.md) | Secret management, MCP Hub, OAuth setup |
| [Deployment](./docs/deployment.md) | Docker, Kubernetes deployment |
| [Benchmarks](./docs/benchmarks.md) | Performance metrics |
| [Quantum Primer](./docs/quantum_primer.md) | Quantum ML theory |

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

## KeepSave Integration

MedQCNN integrates with [KeepSave](https://github.com/santapong/KeepSave) for secure secret management, environment promotion, and centralized MCP server hosting. This eliminates hardcoded credentials in `.env` files and enables safe configuration management across deployment stages.

### Integration Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KeepSave Platform                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Secret Vault │  │  MCP Server Hub  │  │  OAuth 2.0 Provider  │  │
│  │              │  │                  │  │                      │  │
│  │ DATABASE_URL │  │  Registers       │  │  Issues JWT tokens   │  │
│  │ JWT_SECRET   │  │  MedQCNN MCP     │  │  for MedQCNN API     │  │
│  │ API_KEYS     │  │  server as a     │  │  authentication      │  │
│  │ KAFKA_CREDS  │  │  discoverable    │  │                      │  │
│  │ OPENAI_KEY   │  │  tool            │  │                      │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬───────────┘  │
└─────────┼───────────────────┼────────────────────────┼──────────────┘
          │                   │                        │
          ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MedQCNN Service                              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ API Server   │  │  MCP Server      │  │  Training Pipeline   │  │
│  │ (Litestar)   │  │  (diagnose,      │  │  (Trainer, data      │  │
│  │              │  │   model_info,    │  │   loaders, eval)     │  │
│  │  Secrets     │  │   list_datasets) │  │                      │  │
│  │  loaded at   │  │                  │  │  Secrets loaded      │  │
│  │  startup     │  │  Discoverable    │  │  per environment     │  │
│  │  from vault  │  │  via Hub         │  │  (alpha/uat/prod)    │  │
│  └──────────────┘  └──────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### What KeepSave Provides for MedQCNN

| Feature | Benefit |
|---------|---------|
| **Encrypted Secret Vault** | Store `DATABASE_URL`, `JWT_SECRET_KEY`, `KAFKA_BOOTSTRAP_SERVERS`, `OPENAI_API_KEY` securely with AES-256-GCM encryption instead of `.env` files |
| **Environment Promotion** | Promote MedQCNN configs through Alpha → UAT → PROD with diff review and audit trail |
| **MCP Server Hub** | Register MedQCNN's MCP server in KeepSave's marketplace for centralized discovery and management |
| **OAuth 2.0 Provider** | Use KeepSave as the identity provider for MedQCNN's API authentication |
| **API Key Management** | Issue scoped, time-limited API keys for agents accessing MedQCNN through KeepSave |
| **Python SDK** | Fetch secrets at runtime using the KeepSave Python SDK instead of reading `.env` files |

### Setup: Fetch Secrets from KeepSave

#### 1. Install the KeepSave Python SDK

```bash
pip install keepsave
# or add to pyproject.toml
```

#### 2. Store MedQCNN Secrets in KeepSave

```bash
# Using KeepSave CLI
keepsave login --api-url http://localhost:8080 --email admin@example.com

# Create a project for MedQCNN
keepsave push --project medqcnn --env alpha --file .env
```

Or via the KeepSave API:

```bash
# Create project
curl -X POST http://localhost:8080/api/v1/projects \
  -H "Authorization: Bearer <token>" \
  -d '{"name": "medqcnn", "description": "MedQCNN Quantum Diagnostics"}'

# Store secrets
curl -X POST http://localhost:8080/api/v1/projects/<id>/secrets \
  -H "Authorization: Bearer <token>" \
  -d '{"key": "DATABASE_URL", "value": "postgresql://medqcnn:medqcnn@localhost:5432/medqcnn", "environment": "alpha"}'

curl -X POST http://localhost:8080/api/v1/projects/<id>/secrets \
  -H "Authorization: Bearer <token>" \
  -d '{"key": "JWT_SECRET_KEY", "value": "your-strong-random-secret", "environment": "alpha"}'
```

#### 3. Load Secrets at Runtime

```python
from keepsave import KeepSaveClient
import os

# Initialize KeepSave client
ks = KeepSaveClient(
    base_url=os.environ.get("KEEPSAVE_URL", "http://localhost:8080"),
    api_key=os.environ.get("KEEPSAVE_API_KEY"),
)

# Fetch secrets for the current environment
env = os.environ.get("MEDQCNN_ENV", "alpha")
secrets = ks.list_secrets("medqcnn-project-id", env)

# Inject into environment
for secret in secrets:
    os.environ[secret["key"]] = secret["value"]

# Now start MedQCNN normally
from medqcnn.api.server import create_app
app = create_app()
```

#### 4. Promote Configs Between Environments

```bash
# Preview what will change
keepsave promote --project medqcnn --from alpha --to uat --dry-run

# Promote alpha -> uat
keepsave promote --project medqcnn --from alpha --to uat

# Promote uat -> prod (may require approval)
keepsave promote --project medqcnn --from uat --to prod
```

### Setup: Register MedQCNN MCP Server in KeepSave Hub

Register MedQCNN's MCP server so AI agents can discover and use it through KeepSave's unified gateway:

```bash
# Register MedQCNN MCP server
curl -X POST http://localhost:8080/api/v1/mcp/servers \
  -H "Authorization: Bearer <token>" \
  -d '{
    "name": "medqcnn",
    "description": "Hybrid Quantum-Classical CNN for Medical Image Diagnostics",
    "github_url": "https://github.com/santapong/MedQCNN",
    "github_branch": "main",
    "entry_command": "uv run python scripts/mcp_server.py",
    "transport": "stdio",
    "is_public": true,
    "env_mappings": {
      "DATABASE_URL": "medqcnn/alpha/DATABASE_URL",
      "JWT_SECRET_KEY": "medqcnn/alpha/JWT_SECRET_KEY",
      "CHECKPOINT_PATH": "medqcnn/alpha/CHECKPOINT_PATH"
    }
  }'
```

The `env_mappings` field tells KeepSave to inject secrets as environment variables when launching the MCP server. Agents calling MedQCNN tools through the gateway never see the raw credentials.

### Setup: OAuth 2.0 Authentication

Replace MedQCNN's built-in JWT auth with KeepSave as the identity provider:

```bash
# Register MedQCNN as an OAuth client in KeepSave
curl -X POST http://localhost:8080/api/v1/oauth/clients \
  -H "Authorization: Bearer <token>" \
  -d '{
    "name": "MedQCNN API",
    "redirect_uris": ["http://localhost:8000/auth/callback"],
    "scopes": ["read", "write"],
    "grant_types": ["authorization_code", "client_credentials"]
  }'
```

Then validate tokens in MedQCNN against KeepSave's OIDC discovery endpoint:

```bash
# MedQCNN can verify tokens using KeepSave's well-known endpoint
curl http://localhost:8080/.well-known/openid-configuration
```

### Docker Compose (Full Stack)

Run MedQCNN + KeepSave together:

```yaml
# docker-compose.override.yml
services:
  keepsave-api:
    image: santapong/keepsave:latest
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://keepsave:keepsave@keepsave-db:5432/keepsave
      MASTER_KEY: ${MASTER_KEY}
      JWT_SECRET: ${JWT_SECRET}

  keepsave-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: keepsave
      POSTGRES_PASSWORD: keepsave
      POSTGRES_DB: keepsave

  medqcnn-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      KEEPSAVE_URL: http://keepsave-api:8080
      KEEPSAVE_API_KEY: ${KEEPSAVE_API_KEY}
      MEDQCNN_ENV: alpha
    depends_on:
      - keepsave-api
```

### Environment Variables for Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `KEEPSAVE_URL` | `http://localhost:8080` | KeepSave API base URL |
| `KEEPSAVE_API_KEY` | — | KeepSave API key for fetching secrets |
| `MEDQCNN_ENV` | `alpha` | KeepSave environment to load secrets from (`alpha`, `uat`, `prod`) |

For full KeepSave documentation, see [KeepSave README](https://github.com/santapong/KeepSave).

## License

MIT
