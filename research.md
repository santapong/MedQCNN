# MedQCNN Research Plan

This document outlines the research direction for **MedQCNN** — a hybrid
quantum-classical CNN for medical image diagnostics on edge hardware
(Raspberry Pi 5 class, 16–32 GB RAM, 4–8 qubit simulation).

## 1. Research Question

> Can a shallow variational quantum circuit (VQC), fed by a frozen
> ResNet-18 feature projector and OpenCV-enhanced medical images,
> match or outperform a size-matched fully-classical head on small
> medical-imaging benchmarks — while fitting an 8-qubit simulation
> budget on edge hardware?

Sub-questions:

1. How much of the accuracy gap (if any) can classical OpenCV
   pre-processing close on low-contrast modalities?
2. Does amplitude encoding of a 256-D latent vector provide measurable
   benefit over a classical linear head of equal parameter count?
3. Which ansatz depth minimises barren-plateau effects while keeping
   expressivity on 2-class and multi-class MedMNIST subsets?

## 2. Datasets

| Dataset        | Task                        | Classes | Modality     |
|----------------|-----------------------------|---------|--------------|
| BreastMNIST    | Benign vs. malignant        | 2       | Ultrasound   |
| PneumoniaMNIST | Pneumonia detection         | 2       | Chest X-ray  |
| DermaMNIST     | Skin-lesion classification  | 7       | Dermatoscopy |
| PathMNIST      | Colon-tissue classification | 9       | Histopathology |
| OrganAMNIST    | Organ classification        | 11      | Axial CT     |

Raw-image ingest (DICOM, NIfTI) is handled by `medqcnn/data/dicom.py`
and `medqcnn/data/preprocessing.py` so the same pipeline can be applied
to clinical files outside the MedMNIST benchmark.

## 3. Role of OpenCV

OpenCV is used for the classical preprocessing stage that runs **before**
the ResNet backbone:

- **CLAHE** (`medqcnn.data.opencv_ops.clahe`) — local contrast
  enhancement. Primary target: chest X-ray and ultrasound, where global
  histogram equalization tends to wash out lesion boundaries.
- **Gamma correction** (`gamma`) — non-linear intensity remap; useful
  to compensate for modality-specific acquisition curves.
- **Bilateral denoise** (`bilateral`) — edge-preserving smoothing to
  reduce speckle in ultrasound and Poisson-like noise in low-dose CT
  without blurring lesion edges.
- **Unsharp mask** (`unsharp`) — emphasises fine structures in
  histopathology slides.

These ops are deliberately kept small, deterministic, and parameterised
so they can be toggled in ablation studies and recorded alongside the
training run in the `training_runs` database table.

## 4. Experimental Protocol

1. **Baseline** — ResNet-18 features → linear classifier (no quantum,
   no OpenCV enhancement).
2. **+ OpenCV** — same as baseline but with CLAHE + bilateral applied
   to every input. Isolates the contribution of classical enhancement.
3. **+ Quantum** — ResNet-18 features → amplitude encoding → 4-layer
   hardware-efficient ansatz → Pauli-Z readout. Isolates the
   quantum contribution.
4. **Full hybrid** — OpenCV + ResNet-18 + VQC.

Each configuration is run on every dataset in §2 for 10 epochs (fast
loop) and 50 epochs (reported). Seed fixed at 42 (`medqcnn.config.SEED`).

### Metrics

- Accuracy, macro-F1, AUROC.
- **Edge metrics**: parameter count, RSS peak, p50 / p95 inference
  latency on a single Raspberry Pi 5 core (already captured by the
  `benchmarks` table).
- **Quantum metrics**: gradient variance per layer (barren-plateau
  indicator), trained parameter norm.

## 5. Reproducibility

- Fixed global seed (`medqcnn.config.constants.SEED = 42`).
- Every training run is stored in the `training_runs` table with its
  config JSON, including the OpenCV preprocessing parameters.
- The quantum circuit is simulated via PennyLane `default.qubit`
  (state-vector) — deterministic given the seed.
- Docker image pin: see `Dockerfile` and `docker-compose.yml`.

## 6. Next Research Steps

- [ ] Add noise-model experiments via `pennylane-qiskit` Aer backend
      (depolarising + readout noise at realistic rates).
- [ ] Move from 2-D slice extraction to 3-D NIfTI volume inference
      (majority-vote over axial slices).
- [ ] Grad-CAM-style explainability: backproject gradients from the
      Pauli-Z readout through the classical head to the input image.
- [ ] RAG over a small corpus of radiology reports (ChromaDB) so the
      MCP agent can attach a natural-language rationale to each
      diagnosis.
- [ ] External validation on one non-MedMNIST dataset per modality.

## 7. Clinical-Use Disclaimer

MedQCNN is a research prototype. It is **not** a medical device and
must not be used for clinical decision-making. Predictions are
probabilistic outputs of a machine-learning model trained on public
benchmark data and carry no regulatory approval.
