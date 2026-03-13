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
| **Node A** — Classical | Feature extraction + compression to ℝ²⁵⁶ | `medqcnn/classical/` |
| **Node B** — Quantum | Amplitude encoding + HEA + local Pauli-Z | `medqcnn/quantum/` |
| **Hybrid Model** | End-to-end differentiable pipeline | `medqcnn/model/` |
| **API Server** | REST endpoints for inference | `medqcnn/api/` |
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
| `POST` | `/predict` | Inference (base64 image → diagnosis) |

Example:
```bash
# Health check
curl http://localhost:8000/health

# Predict (base64 image)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-encoded-image>"}'
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

The project includes a Next.js web dashboard for interactive diagnosis.

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
- `/model` — View model architecture, parameter counts, and service health

```bash
# Production build
cd frontend && bun run build && bun run start
```

## Docker Deployment

```bash
# Build and run with Kafka
docker compose up -d

# API available at http://localhost:8000
# Kafka broker at localhost:9092
```

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
│   ├── mcp/                 # MCP server for AI agents
│   └── utils/               # Logging, device management
├── scripts/                 # CLI scripts (demo, train, serve, mcp)
├── frontend/                # Next.js web dashboard (Bun)
├── tests/                   # Unit tests
├── notebooks/               # Educational notebooks
├── Dockerfile               # Multi-stage container
├── docker-compose.yml       # API + Kafka orchestration
└── GEMINI.md                # Full architecture spec
```

## Hardware Constraints

- **Qubits:** 8 max (256-dim latent space)
- **Ansatz layers:** 4 (shallow — barren plateau mitigation)
- **Target:** CPU-only, 16–32 GB RAM, Raspberry Pi 5 cluster
- **Simulation:** State-vector via PennyLane `default.qubit`

## Tech Stack

| Layer | Technology |
|-------|------------|
| Classical Vision | PyTorch, torchvision (ResNet-18) |
| Quantum Circuit | PennyLane (TorchLayer, backprop) |
| API Server | Litestar, Uvicorn |
| Agent Protocol | MCP (Model Context Protocol) |
| Message Broker | Apache Kafka |
| Containerization | Docker, Docker Compose |
| Frontend | Next.js 16, TypeScript, Tailwind CSS v4 |
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
