"""Post-hoc inference utilities (calibration, conformal, OOD).

These tools attach to a *trained* HybridQCNN checkpoint via sidecar
JSON files. Nothing here mutates the model weights or the training
contract — keeping the trust-layer artefacts decoupled from the
torch.save format avoids schema migrations and lets you re-calibrate
or re-fit a conformal predictor without touching the checkpoint.
"""

from medqcnn.inference.conformal import ConformalPredictor
from medqcnn.inference.ood import OODDetector

__all__ = ["ConformalPredictor", "OODDetector"]
