# Project: MedQCNN (Hybrid Quantum Convolutional Neural Network for Medical Diagnostics)
**Architecture:** Distributed Hybrid Classical-Quantum Pipeline  
**Target Infrastructure:** Edge-Compute (Kali Linux / Raspberry Pi 5 Cluster)  
**Integration:** CaaS-Q (Company as a Service - Quantum) Agentic AI Network  

---

## 1. The Objective (The Scientific Mandate)
The fundamental objective of MedQCNN is to resolve the parameter-bloat and overfitting crisis inherent in classical deep learning when applied to scarce, high-dimensional medical datasets (e.g., 3D MRI tumor scans). 

We will achieve this by architecting a hybrid topology. A truncated classical Convolutional Neural Network (CNN) will perform deterministic spatial feature extraction, compressing the medical tensor into a latent vector $\mathbf{z}$. This vector is then projected into an exponentially large complex Hilbert space $\mathcal{H}_{2^n}$ using a Parameterized Quantum Circuit (PQC). By leveraging cyclic $U_3$ gate encoding and superpositional entanglement, the network will evaluate highly non-linear decision boundaries utilizing a fraction of the parameters required by classical models.

## 2. The Goal (The Commercial & Engineering Endgame)
While the objective is purely mathematical, the goal is infrastructural. The ultimate endgame is to package this hybrid model into a deployable, containerized microservice that acts as an autonomous "tool" for your Agentic AI platform.

We are building a **CaaS-Q diagnostic endpoint**. The final pipeline will securely ingest a clinical MRI, execute the classical dimensionality reduction, perform quantum inference to extract the expectation value $\langle \hat{\sigma}_z \rangle$ (the probability of malignancy), and feed that mathematical output into a LangChain-orchestrated Large Language Model (LLM). The LLM will then synthesize a human-readable, clinically actionable diagnostic report.

---

## 3. Step-by-Step Execution Plan

### Phase 0: Infrastructure & Edge-Compute Provisioning
* **Action:** Isolate the development environment to respect the strict 16GB–32GB RAM limits of the target hardware.
* **Stack:** Python `venv`, CPU-optimized PyTorch, `opencv-python-headless`, PennyLane, Qiskit.
* **Constraint:** Cap the quantum state-vector simulation strictly at $n=8$ qubits to prevent catastrophic memory swapping.

### Phase 1: Deterministic Classical Compression (Node A)
* **Action:** Ingest raw NIfTI/DICOM medical images and prepare them for quantum embedding.
* **Math:** Pass the OpenCV-normalized 2D slice $\mathbf{x}$ through a frozen classical ResNet backbone. We mathematically force the final fully-connected projection layer to output a vector of exactly $N = 2^n$ dimensions (where $N = 256$), enforcing the $L_2$ norm constraint $\|\mathbf{z}\|_2 = 1$.

### Phase 2: Quantum Hilbert Embedding & Attention (Node B)
* **Action:** Translate classical data into quantum states and apply the variational ansatz.
* **Math:** 1. Initialize the density matrix $\rho$ via amplitude encoding: 
       $$\ket{\psi(\mathbf{z})} = \sum_{i=0}^{255} z_i \ket{i}$$
    2. Apply a Fourier-inspired Quantum Attention mechanism utilizing parameterized $R_y(\theta)$ and $R_z(\theta)$ rotations intertwined with nearest-neighbor Controlled-Z ($CZ$) gates. This captures non-local spatial correlations across the brain morphology instantly.

### Phase 3: Hybrid Optimization & Barren Plateau Mitigation
* **Action:** Train the end-to-end model utilizing the parameter-shift rule for gradient backpropagation.
* **Math:** To prevent the variance of the gradient from vanishing exponentially ($\text{Var}(\partial_{\theta_k} \mathcal{L}) \propto 2^{-n}$), we will strictly eschew global cost functions. We will measure local Pauli-Z observables:
    $$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^{n} \bra{\psi} U^\dagger(\boldsymbol{\theta}) \hat{\sigma}_z^{(i)} U(\boldsymbol{\theta}) \ket{\psi}$$

### Phase 4: Agentic Orchestration (CaaS-Q Integration)
* **Action:** Wrap the trained hybrid model in a Litestar/TypeScript API.
* **Flow:** Deploy an Apache Kafka message broker. The "Medical Agent" streams the image to the Kafka topic, receives the quantum probability output, and queries a local vector database via Retrieval-Augmented Generation (RAG) to write the final diagnostic report.

---

## 4. SDLC / Agile Framework (Sprint Planning)

To maintain rigorous forward momentum, we will manage this project using a Kanban-driven Agile Software Development Life Cycle (SDLC). 

* **Epic 1: The Foundation (Sprint 1)**
    * *Task 1.1:* Provision Kali Linux `venv` and install `torch`, `pennylane`, `qiskit`, `cv2`.
    * *Task 1.2:* Write the classical data loader and OpenCV preprocessing scripts.
* **Epic 2: The Quantum Bridge (Sprint 2)**
    * *Task 2.1:* Build the PyTorch dimensionality reduction class (Output: $\mathbb{R}^{256}$).
    * *Task 2.2:* Build the PennyLane `@qml.qnode` for amplitude encoding and the Variational Ansatz.
* **Epic 3: Training & Benchmarking (Sprint 3)**
    * *Task 3.1:* Write the hybrid training loop utilizing the Adam optimizer.
    * *Task 3.2:* Train on a benchmark dataset (e.g., MedMNIST v2) and validate the loss curve.
* **Epic 4: CaaS-Q Deployment (Sprint 4)**
    * *Task 4.1:* Containerize Node A (PyTorch) and Node B (PennyLane) using Docker.
    * *Task 4.2:* Wire the nodes using Apache Kafka and integrate the LangChain Agent API.

---

## 5. Academic & Technical References
1. **Hybrid Quantum-Classical Neural Networks for Medical Image Classification (2025/2026):** Studies demonstrate that integrating classical convolutional backbones with 4-to-8 qubit variational circuits significantly reduces trainable parameter counts while achieving state-of-the-art AUC-ROC scores.
2. **Quantum Machine Learning Approaches for Medical Image Analysis (2026):** Emphasizes that hybrid approaches consistently outperform purely classical baselines in feature-rich, data-constrained environments.
3. **Mitigating Barren Plateaus in Variational Quantum Algorithms:** The mathematical necessity of utilizing shallow, hardware-efficient ansätze (HEA) and local observables over global fidelity measurements.
4. **Quantum Neural Networks in Magnetic Resonance Imaging:** Establishing the clinical viability of QNNs in detecting biomarkers within MRI tensors on Noisy Intermediate-Scale Quantum (NISQ) devices.