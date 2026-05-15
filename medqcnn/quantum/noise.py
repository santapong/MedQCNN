"""
Aer noise models for NISQ-credible VQC training and evaluation.

Provides:
  - `NoiseConfig`: a frozen dataclass capturing depolarising and readout
    noise rates.
  - `NOISE_PRESETS`: four named operating points (none / low / medium / high)
    drawn from the rate ranges reported in arXiv:2604.11541 and the
    QCQ-CNN robustness study (Sci. Reports 2025).
  - `build_noise_model`: constructs a `qiskit_aer.noise.NoiseModel` that
    can be passed to PennyLane's `qiskit.aer` device.

The module deliberately covers only the two error channels that dominate
on superconducting hardware in the regimes accessible to a 4–8 qubit
shallow ansatz: depolarising on 1- and 2-qubit gates, plus symmetric
readout flip on measurement. Amplitude/phase damping and crosstalk are
intentionally out of scope for this first harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NoiseConfig:
    """Two-channel noise specification for the Aer simulator.

    Attributes:
        depolarising_p: Per-gate depolarising probability applied to
            single-qubit (RY, RZ) and two-qubit (CZ, CX) gates.
        readout_p: Symmetric bit-flip probability on measurement.
    """

    depolarising_p: float = 0.0
    readout_p: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("depolarising_p", self.depolarising_p),
            ("readout_p", self.readout_p),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")

    @property
    def is_noiseless(self) -> bool:
        return self.depolarising_p == 0.0 and self.readout_p == 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NOISE_PRESETS: dict[str, NoiseConfig] = {
    "none": NoiseConfig(depolarising_p=0.0, readout_p=0.0),
    "low": NoiseConfig(depolarising_p=0.001, readout_p=0.01),
    "medium": NoiseConfig(depolarising_p=0.005, readout_p=0.02),
    "high": NoiseConfig(depolarising_p=0.01, readout_p=0.03),
}
"""Named noise operating points used by the sweep harness.

`high` sits near the depolarising rate at which the QCQ-CNN study
(Sci. Reports 2025) reports collapse for small ansatz heads. `low`
matches calibration data for current cloud superconducting backends.
"""


def get_preset(name: str) -> NoiseConfig:
    """Look up a noise preset by name with a useful error message."""
    try:
        return NOISE_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(NOISE_PRESETS))
        raise KeyError(f"Unknown noise preset {name!r}. Valid: {valid}.") from exc


def build_noise_model(config: NoiseConfig):
    """Build a qiskit-aer NoiseModel from a `NoiseConfig`.

    Returns ``None`` for the noiseless case so callers can branch on it
    cheaply and skip routing through the noisy device.

    The depolarising channel is added to all qubits for the standard
    PennyLane → Qiskit transpilation gate set (RY/RZ/U1/U2/U3/CZ/CX).
    Readout error is symmetric: ``P(0|1) = P(1|0) = readout_p``.
    """
    if config.is_noiseless:
        return None

    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise_model = NoiseModel()

    if config.depolarising_p > 0.0:
        one_q_error = depolarizing_error(config.depolarising_p, 1)
        two_q_error = depolarizing_error(config.depolarising_p, 2)
        noise_model.add_all_qubit_quantum_error(
            one_q_error, ["ry", "rz", "u1", "u2", "u3"]
        )
        noise_model.add_all_qubit_quantum_error(two_q_error, ["cz", "cx"])

    if config.readout_p > 0.0:
        p = config.readout_p
        readout = ReadoutError([[1.0 - p, p], [p, 1.0 - p]])
        noise_model.add_all_qubit_readout_error(readout)

    return noise_model
