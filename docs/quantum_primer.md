# Quantum Computing Primer for MedQCNN

A concise guide to the quantum computing concepts used in this project.

## 1. Qubits

A classical bit is 0 or 1. A **qubit** can be in a superposition of both:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $|\alpha|^2 + |\beta|^2 = 1$.

**In MedQCNN:** We use 8 qubits, which can represent $2^8 = 256$ states simultaneously — matching our latent vector dimension.

## 2. Amplitude Encoding

We map a classical vector $\mathbf{z} \in \mathbb{R}^{2^n}$ (with $\|\mathbf{z}\|_2 = 1$) into a quantum state:

$$|\psi(\mathbf{z})\rangle = \sum_{i=0}^{2^n-1} z_i |i\rangle$$

Each component $z_i$ becomes an **amplitude** of basis state $|i\rangle$.

**Advantage:** Encodes $2^n$ values using only $n$ qubits — exponential compression.

**In MedQCNN:** 256 latent features → 8 qubits (`medqcnn/quantum/encoding.py`).

## 3. Quantum Gates

### Single-Qubit Rotations
- **$R_y(\theta)$**: Rotation around Y-axis — mixes $|0\rangle$ and $|1\rangle$
- **$R_z(\theta)$**: Rotation around Z-axis — adds relative phase

These are the trainable parameters that the optimizer updates.

### Two-Qubit Gates
- **$CZ$ (Controlled-Z)**: Creates **entanglement** between two qubits

**In MedQCNN:** Each ansatz layer applies $R_y + R_z$ per qubit, then $CZ$ gates in a ring topology (`medqcnn/quantum/ansatz.py`).

## 4. Variational Ansatz (HEA)

The **Hardware-Efficient Ansatz** is a parameterized circuit:

```
Layer 1:  Ry(θ₁) Rz(θ₂) ──CZ── Ry(θ₃) Rz(θ₄) ──CZ── ...
Layer 2:  Ry(θ₅) Rz(θ₆) ──CZ── Ry(θ₇) Rz(θ₈) ──CZ── ...
...
```

Each layer has $2n$ parameters ($R_y$ and $R_z$ per qubit).
Total: `n_layers × n_qubits × 2` parameters.

**In MedQCNN:** 4 layers × 8 qubits × 2 gates = **64 parameters**.

## 5. Measurement (Pauli-Z Observable)

After the circuit evolves the state, we measure each qubit:

$$\langle\sigma_z^{(i)}\rangle \in [-1, 1]$$

This gives us $n$ expectation values — one per qubit.

**Why local observables?** Global measurements (like fidelity) cause **barren plateaus**.

## 6. Barren Plateaus

The gradient of a random quantum circuit vanishes exponentially:

$$\text{Var}\left(\frac{\partial \mathcal{L}}{\partial \theta_k}\right) \propto 2^{-n}$$

**Mitigations in MedQCNN:**
1. **Local observables** (per-qubit $\sigma_z$) instead of global cost
2. **Shallow circuit** (4 layers, not 20+)
3. **Small qubit count** (8 max — $2^{-8} = 0.004$ is manageable)

## 7. Parameter-Shift Rule

Quantum circuit gradients are computed via:

$$\frac{\partial f}{\partial \theta} = \frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$$

This works on real quantum hardware (no backpropagation needed).

**In MedQCNN:** Simulator uses `backprop` (faster); hardware would use `parameter-shift` (`medqcnn/quantum/qnode.py`).

## 8. Why Quantum for Medical Imaging?

| Problem | Classical | Quantum |
|---------|-----------|---------|
| FC layer (256→256) | 65,792 params | 64 params (8 qubits × 4 layers × 2) |
| Feature space | Euclidean $\mathbb{R}^{256}$ | Hilbert $\mathcal{H}_{256}$ (complex) |
| Expressibility | Polynomial | Exponential (superposition + entanglement) |
| Overfitting risk | High (data-scarce) | Low (far fewer parameters) |

The quantum circuit acts as a highly expressive but parameter-efficient classifier — ideal for medical datasets with hundreds, not millions, of samples.
