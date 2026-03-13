# API Reference

## REST API (Litestar)

Base URL: `http://localhost:8000`

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "medqcnn",
  "version": "0.1.0"
}
```

### GET /info

Model architecture and parameter details.

**Response:**
```json
{
  "n_qubits": 4,
  "latent_dim": 16,
  "n_ansatz_layers": 4,
  "n_classes": 2,
  "trainable_params": {
    "projector": 271920,
    "quantum": 32,
    "classifier": 226,
    "total": 272178
  },
  "device": "cpu"
}
```

### POST /predict

Run quantum-classical inference on a medical image.

**Request:**
```json
{
  "image_base64": "<base64-encoded image>",
  "return_probabilities": true
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "Benign",
  "confidence": 0.5234,
  "probabilities": [0.5234, 0.4766],
  "quantum_expectation_values": [0.1234, -0.0567, 0.2345, -0.1890]
}
```

**Example:**
```bash
# Encode image
IMAGE_B64=$(base64 -w 0 data/test_sample.png)

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\"}"
```

### POST /predict/batch

Run inference on multiple images in a single request (max 100 images, 250 MB total).

**Request:**
```json
{
  "images": ["<base64-image-1>", "<base64-image-2>"]
}
```

**Response:**
```json
{
  "results": [
    {"prediction": 0, "label": "Benign", "confidence": 0.82, "probabilities": [0.82, 0.18], "quantum_expectation_values": [...]},
    {"prediction": 1, "label": "Malignant", "confidence": 0.91, "probabilities": [0.09, 0.91], "quantum_expectation_values": [...]}
  ],
  "total": 2,
  "successful": 2,
  "failed": 0,
  "summary": {"benign": 1, "malignant": 1, "avg_confidence": 0.865},
  "errors": null
}
```

### GET /predictions

List prediction history with filtering and pagination.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 50 | Items per page (max 200) |
| `label` | string | — | Filter by label (Benign/Malignant) |
| `confidence_min` | float | — | Minimum confidence threshold |
| `filename` | string | — | Search by filename (partial match) |

**Response:**
```json
{
  "items": [{"id": 1, "image_filename": "scan.png", "label": "Benign", "confidence": 0.82, ...}],
  "total": 42,
  "offset": 0,
  "limit": 50
}
```

### GET /predictions/{id}

Get a single prediction by ID.

### GET /training-runs

List training runs with metrics and history.

**Response:**
```json
{
  "items": [
    {
      "id": 1, "dataset": "breastmnist", "n_qubits": 4, "epochs": 10,
      "final_train_acc": 0.85, "final_val_acc": 0.78,
      "history": {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]},
      "benchmarks": [{"metric_name": "params_total", "metric_value": 8706}]
    }
  ],
  "total": 3, "offset": 0, "limit": 50
}
```

### GET /training-runs/{id}

Get a single training run by ID.

### GET /benchmarks

List benchmark metrics, optionally filtered by training run.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `training_run_id` | int | — | Filter by training run ID |
| `offset` | int | 0 | Pagination offset |
| `limit` | int | 100 | Items per page (max 500) |

---

## MCP Server

Transport: `stdio`

### Tool: diagnose

Run quantum inference on a medical image file.

| Param | Type | Description |
|-------|------|-------------|
| `image_path` | string | Absolute path to the image file |

**Returns:** JSON with prediction, confidence, probabilities, quantum expectation values.

### Tool: model_info

Get model architecture and parameter details.

**Returns:** JSON with architecture config, parameter counts, device info.

### Tool: list_datasets

List available MedMNIST benchmark datasets.

**Returns:** JSON catalog of supported medical imaging datasets.

### Configuration

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

---

## LangChain Tools

### quantum_diagnose

```python
from medqcnn.agent.tools import quantum_diagnose
result = quantum_diagnose.invoke("/path/to/image.png")
```

### get_model_info

```python
from medqcnn.agent.tools import get_model_info
info = get_model_info.invoke("")
```

### list_medical_datasets

```python
from medqcnn.agent.tools import list_medical_datasets
datasets = list_medical_datasets.invoke("")
```

### Building an Agent

```python
from langchain_openai import ChatOpenAI
from medqcnn.agent.agent import create_agent_executor

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent_executor(llm)

result = agent.invoke({"messages": [
    HumanMessage(content="Analyze this scan: /data/breast_scan.png")
]})
```
