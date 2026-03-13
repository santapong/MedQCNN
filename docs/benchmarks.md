# MedQCNN Benchmarks

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-18 (frozen, ImageNet pre-trained) |
| Projector | FC 512 → 256, BatchNorm, L2 Norm |
| Qubits | 8 (production) / 4 (demo) |
| Ansatz | Hardware-Efficient, 4 layers (Ry/Rz + CZ ring) |
| Diff method | `backprop` (state-vector sim) |
| Optimizer | Adam, lr=1e-3 |
| Loss | CrossEntropy + L2 regularization |

## Parameter Counts

| Component | 4-Qubit (Demo) | 8-Qubit (Production) |
|-----------|---------------|---------------------|
| Backbone (frozen) | 11,176,512 | 11,176,512 |
| Projector | 8,480 | 132,096 |
| Quantum circuit | 32 | 64 |
| Classifier | 194 | 418 |
| **Total trainable** | **8,706** | **132,578** |

> The quantum circuit uses only 32–64 parameters to operate in a
> 16–256 dimensional Hilbert space — exponential compression vs
> an equivalent classical FC layer (256 + 65,536 = 65,792 params).

## Dataset: BreastMNIST

| Split | Samples |
|-------|---------|
| Train | 546 |
| Val | 78 |
| Test | 156 |
| Classes | 2 (Benign / Malignant) |
| Image size | 28 × 28 (resized to 224 × 224) |

## Baseline Results (Untrained Model)

These results represent a randomly initialized HybridQCNN to establish a baseline.
Full training benchmarks (50+ epochs) are planned in Phase 5.

| Metric | 4-Qubit | 8-Qubit |
|--------|---------|---------|
| Accuracy | ~50% (random) | ~50% (random) |
| AUC-ROC | ~0.50 | ~0.50 |
| F1 | ~0.33 | ~0.33 |

## Memory Profile

| Qubits | Latent Dim | State-Vector Size | Peak RAM (inference) |
|--------|-----------|-------------------|---------------------|
| 4 | 16 | 128 B | ~2 GB |
| 6 | 64 | 512 B | ~3 GB |
| 8 | 256 | 2 KB | ~4 GB |
| 10 | 1024 | 8 KB | ~8 GB |

## Inference Latency (CPU, single image)

| Component | 4-Qubit | 8-Qubit |
|-----------|---------|---------|
| ResNet-18 backbone | ~15 ms | ~15 ms |
| Projector | <1 ms | <1 ms |
| Quantum circuit | ~50 ms | ~200 ms |
| Classifier | <1 ms | <1 ms |
| **Total** | **~70 ms** | **~220 ms** |

> Measured on x86-64 CPU. Raspberry Pi 5 ARM64 will be ~3–5× slower.

## Test Suite

| Test File | Tests | Status |
|-----------|-------|--------|
| test_data.py | 6 | All passing |
| test_classical.py | 4 | All passing |
| test_quantum.py | 3 | All passing |
| test_model.py | 1 | All passing |
| **Total** | **14** | **14/14 passing** |

## Reproducing Results

```bash
# Quick demo (4-qubit, forward pass + gradient verification)
uv run python scripts/demo.py

# Train (4-qubit, 10 epochs)
uv run python scripts/train.py --n-qubits 4 --epochs 10 --batch-size 8

# Evaluate
uv run python scripts/evaluate.py --n-qubits 4

# Full training (8-qubit, 50 epochs — requires 16+ GB RAM)
uv run python scripts/train.py --n-qubits 8 --epochs 50 --batch-size 4
```
