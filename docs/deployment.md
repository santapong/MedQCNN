# Deployment Guide

## Local Development

### Prerequisites
- Python 3.11+
- `uv` package manager (v0.6+)

### Setup
```bash
# Clone and install
git clone <repo-url>
cd MedQCNN
uv sync

# Install dev tools
uv sync --extra dev

# Verify
uv run python main.py
```

### Running Services
```bash
# Demo (end-to-end forward pass)
uv run python scripts/demo.py

# Training
uv run python scripts/train.py --epochs 10 --n-qubits 4

# REST API
uv run python scripts/serve.py --port 8000

# MCP Server
uv run python scripts/mcp_server.py

# Tests
uv run python -m pytest tests/ -v
```

---

## Docker Deployment

### Build
```bash
docker build -t medqcnn:latest .
```

### Run (API only)
```bash
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints medqcnn:latest
```

### Run with Kafka
```bash
docker compose up -d
```

This starts:
- **medqcnn-api** on port 8000
- **Kafka** (KRaft mode) on port 9092

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Edge Deployment (Raspberry Pi 5)

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 32 GB SD | 64 GB SSD |
| OS | Raspberry Pi OS 64-bit | Kali Linux ARM64 |
| Python | 3.11 | 3.11 |

### Installation
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone project
git clone <repo-url>
cd MedQCNN

# Install (CPU-only PyTorch is default)
uv sync

# Verify memory
uv run python main.py
# Should show: RAM: X.X / 8.0 GB
```

### Running on Pi 5
```bash
# Use 4 qubits for Pi 5 (16-dim latent, fast)
uv run python scripts/train.py --n-qubits 4 --batch-size 4

# Use 8 qubits only on 16GB+ RAM models
uv run python scripts/train.py --n-qubits 8 --batch-size 2
```

### Memory Guidelines
| Qubits | Latent Dim | State-Vector Size | Min RAM |
|--------|-----------|-------------------|---------|
| 4 | 16 | 128 B | 4 GB |
| 6 | 64 | 512 B | 8 GB |
| 8 | 256 | 2 KB | 16 GB |
| 10 | 1024 | 8 KB | 32 GB |
| 12+ | 4096+ | 32+ KB | ❌ Not supported |

---

## Configuration

### settings/settings.json
```json
{
  "quantum": {
    "n_qubits": 8,
    "n_ansatz_layers": 4,
    "diff_method": "backprop"
  },
  "classical": {
    "backbone": "resnet18",
    "image_size": 224,
    "freeze_backbone": true
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "default_dataset": "breastmnist"
  },
  "hardware": {
    "target_device": "cpu",
    "max_ram_gb": 32
  }
}
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MEDQCNN_QUBITS` | 8 | Override qubit count |
| `MEDQCNN_CHECKPOINT` | None | Checkpoint path for API |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Kafka broker |
