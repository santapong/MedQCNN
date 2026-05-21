"""Tests for split-conformal APS predictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from medqcnn.inference.conformal import ConformalPredictor


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


def _synthetic_probs(n: int, c: int, noise: float = 0.3):
    """Generate (probs, labels) with a known imperfect classifier."""
    labels = np.random.randint(0, c, size=n)
    one_hot = np.eye(c)[labels]
    logits = one_hot * 4.0 + np.random.randn(n, c) * noise * 4.0
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels


class TestConformalPredictor:
    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            ConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalPredictor(alpha=1.0)

    def test_predict_before_calibrate_raises(self):
        cp = ConformalPredictor(alpha=0.1)
        with pytest.raises(RuntimeError):
            cp.predict_set(np.ones((1, 2)) * 0.5)

    def test_marginal_coverage_holds_on_synthetic_iid(self):
        alpha = 0.1
        probs_cal, labels_cal = _synthetic_probs(n=1000, c=4)
        probs_test, labels_test = _synthetic_probs(n=2000, c=4)

        cp = ConformalPredictor(alpha=alpha)
        cp.calibrate(probs_cal, labels_cal)

        sets = cp.predict_set(probs_test)
        covered = sum(int(int(labels_test[i]) in s) for i, s in enumerate(sets))
        empirical = covered / len(labels_test)
        # Marginal coverage guarantee is ≥ 1 - alpha in expectation;
        # for n=2000 the finite-sample slack is small. Allow 3pp slack.
        assert empirical >= (1 - alpha) - 0.03

    def test_sets_are_nonempty_and_sorted(self):
        probs, labels = _synthetic_probs(n=500, c=4)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(probs, labels)
        sets = cp.predict_set(probs[:50])
        for s in sets:
            assert len(s) >= 1
            assert list(s) == sorted(s)
            assert all(0 <= i < 4 for i in s)

    def test_abstains_matches_set_cardinality(self):
        probs, labels = _synthetic_probs(n=500, c=4)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(probs, labels)
        sets = cp.predict_set(probs[:20])
        abstains = cp.abstains(probs[:20])
        assert abstains == [len(s) != 1 for s in sets]

    def test_tighter_alpha_yields_larger_sets(self):
        probs, labels = _synthetic_probs(n=1000, c=4)
        cp_loose = ConformalPredictor(alpha=0.2)
        cp_tight = ConformalPredictor(alpha=0.01)
        cp_loose.calibrate(probs, labels)
        cp_tight.calibrate(probs, labels)
        loose_sizes = [len(s) for s in cp_loose.predict_set(probs)]
        tight_sizes = [len(s) for s in cp_tight.predict_set(probs)]
        assert sum(tight_sizes) >= sum(loose_sizes)

    def test_save_load_round_trip(self, tmp_path: Path):
        probs, labels = _synthetic_probs(n=200, c=3)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(probs, labels)
        path = tmp_path / "conformal.json"
        cp.save(path)

        round_tripped = ConformalPredictor.load(path)
        assert round_tripped.alpha == cp.alpha
        assert round_tripped.qhat == cp.qhat
        assert round_tripped.n_calibration == cp.n_calibration
        assert round_tripped.n_classes == cp.n_classes

        payload = json.loads(path.read_text())
        assert "qhat" in payload and "alpha" in payload

    def test_accepts_torch_tensors(self):
        probs, labels = _synthetic_probs(n=200, c=3)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(torch.from_numpy(probs), torch.from_numpy(labels))
        sets = cp.predict_set(torch.from_numpy(probs[:10]))
        assert len(sets) == 10
