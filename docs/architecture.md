# MedQCNN Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CaaS-Q Platform                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────┐    ┌───────────────────────┐  │
│  │  LangChain   │───▶│  Kafka   │───▶│    MedQCNN Service    │  │
│  │  Agent/LLM   │◀───│  Broker  │◀───│                       │  │
│  └──────────────┘    └──────────┘    │  ┌───────────────┐    │  │
│         │                            │  │   Node A       │    │  │
│         ▼                            │  │  (Classical)   │    │  │
│  ┌──────────────┐                    │  │  ResNet-18 →   │    │  │
│  │  Clinical    │                    │  │  FC → L2 Norm  │    │  │
│  │  Report      │                    │  └───────┬───────┘    │  │
│  └──────────────┘                    │          │            │  │
│                                      │          ▼            │  │
│                                      │  ┌───────────────┐    │  │
│                                      │  │   Node B       │    │  │
│                                      │  │  (Quantum)     │    │  │
│                                      │  │  Amp Encode →  │    │  │
│                                      │  │  HEA → ⟨σ_z⟩   │    │  │
│                                      │  └───────┬───────┘    │  │
│                                      │          │            │  │
│                                      │          ▼            │  │
│                                      │  ┌───────────────┐    │  │
│                                      │  │  Classifier    │    │  │
│                                      │  │  FC → Softmax  │    │  │
│                                      │  └───────────────┘    │  │
│                                      └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Map

### Package Structure

| Module | Phase | Description |
|--------|-------|-------------|
| `medqcnn/config/` | All | Constants (`NUM_QUBITS=8`, `LATENT_DIM=256`), settings |
| `medqcnn/data/` | Phase 1 | NIfTI/MedMNIST loaders, OpenCV preprocessing, transforms |
| `medqcnn/classical/` | Phase 1 | Frozen ResNet-18 backbone + FC projector with L2 norm |
| `medqcnn/quantum/` | Phase 2 | Amplitude encoding, HEA ansatz, Pauli-Z observables, QNode |
| `medqcnn/model/` | Phase 1+2 | `HybridQCNN` — end-to-end differentiable nn.Module |
| `medqcnn/training/` | Phase 3 | Trainer, loss, metrics, visualization |
| `medqcnn/utils/` | All | Logging (Rich), device management, seed reproducibility |
| `medqcnn/api/` | Phase 4 | Litestar REST server, Kafka producer/consumer |
| `medqcnn/mcp/` | Phase 4 | MCP server — tools for AI agent integration |
| `medqcnn/agent/` | Phase 4 | LangChain tools + LangGraph ReAct agent |

### Data Flow

```
Medical Image (224×224×1)
    │
    ▼ ClassicalBackbone (frozen ResNet-18)
Feature Vector (512-dim)
    │
    ▼ LatentProjector (FC → BatchNorm → L2 Norm)
Latent Vector z ∈ R^256, ||z||₂ = 1
    │
    ▼ AmplitudeEncoding
Quantum State |ψ(z)⟩ = Σ zᵢ|i⟩  (8 qubits, 256 basis states)
    │
    ▼ HardwareEfficientAnsatz (4 layers × [Ry, Rz, CZ])
Evolved State U(θ)|ψ(z)⟩
    │
    ▼ Per-Qubit PauliZ Measurement
Expectation Values [⟨σ_z^(0)⟩, ..., ⟨σ_z^(7)⟩]  (8 values ∈ [-1,1])
    │
    ▼ Classifier Head (FC → ReLU → Dropout → FC)
Class Logits → Softmax → Prediction
```

### Integration Points

| Interface | Protocol | Port | Description |
|-----------|----------|------|-------------|
| REST API | HTTP | 8000 | `/health`, `/info`, `/predict` |
| MCP Server | stdio | — | `diagnose`, `model_info`, `list_datasets` |
| LangChain | Python | — | `@tool` decorated functions |
| Kafka | TCP | 9092 | `medqcnn.inference.request/result` topics |

## Hardware Constraints

| Parameter | Value | Reason |
|-----------|-------|--------|
| Max qubits | 8 | State-vector simulation: 2^8 = 256 amplitudes ≈ 2KB |
| Max qubits (theoretical) | 16 | 2^16 = 65,536 amplitudes ≈ 512KB (exceeds Pi 5 limits) |
| Ansatz layers | 4 | Shallow to mitigate barren plateaus |
| Target RAM | 16–32 GB | Raspberry Pi 5 cluster constraint |
| Diff method (sim) | `backprop` | Fast for state-vector simulators |
| Diff method (hardware) | `parameter-shift` | NISQ hardware compatible |
