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
