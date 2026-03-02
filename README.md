# MedQCNN: Hybrid Quantum-Classical Convolutional Neural Network for Medical Diagnostics

## Overview

The **MedQCNN** project seeks to solve the classical curse of dimensionality and parameter-bloat in medical image analysis. Traditional Convolutional Neural Networks (CNNs) require millions of parameters to detect morphological anomalies (e.g., gliomas, Alzheimer's atrophy) in volumetric MRI scans, inevitably leading to severe overfitting on scarce, highly-regulated clinical datasets. 

By mapping classically compressed latent vectors into the exponentially large complex Hilbert space $\mathcal{H}_{2^n}$ of a quantum system, MedQCNN leverages quantum superposition and entanglement to evaluate highly non-linear decision boundaries. The objective is to achieve state-of-the-art diagnostic accuracy using an order of magnitude fewer trainable parameters, ensuring high generalization on small datasets.

## Architecture

MedQCNN is designed as a containerized, event-driven "Company as a Service" (CaaS) tool orchestrated for an AI Agent network, composed of three main nodes:

* **Node A (Classical Vision):** PyTorch & OpenCV for preprocessing and dimensionality reduction ($\mathbf{x} \to \mathbf{z} \in \mathbb{R}^{256}$).
* **Node B (Quantum Inference):** PennyLane QNode executing Amplitude Embedding, a Hardware-Efficient Ansatz (HEA), and observable measurement.
* **Node C (Agentic Orchestration):** TypeScript/Litestar & LangChain using a Kafka message broker for routing and a RAG pipeline to generate clinical diagnostic reports.

### Hardware Constraints
Designed to bypass memory limitations on edge devices (e.g., Raspberry Pi 5 Cluster, Kali Linux Workstation), the quantum simulator is strictly capped at **$n = 8$ qubits** (256-dimensional $L_2$-normalized vectors).

## Tech Stack
- **Language:** Python >= 3.11
- **Package Manager:** `uv`
- **Classical Vision:** PyTorch, OpenCV, NiBabel, NumPy, Pandas
- **Quantum Machine Learning:** PennyLane, Qiskit

## Documentation
- See [CHANGELOG.md](CHANGELOG.md) for release history and notable changes.
- See [GEMINI.md](GEMINI.md) for detailed execution roadmap, agent guidelines, and project constraints.
