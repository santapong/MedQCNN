"""Tests for temperature scaling and ECE."""

from __future__ import annotations

import pytest
import torch

from medqcnn.training.calibration import (
    apply_temperature,
    compute_ece,
    fit_temperature,
    reliability_curve,
)


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(0)


def _make_overconfident_logits(n: int = 512, c: int = 4, scale: float = 5.0):
    """Synthetic over-confident classifier: high-magnitude logits with
    label noise so confidence systematically exceeds accuracy."""
    labels = torch.randint(0, c, (n,))
    base = torch.zeros(n, c)
    base[torch.arange(n), labels] = 1.0
    # Flip the argmax for 30% of samples → over-confident but only 70% acc
    flip = torch.rand(n) < 0.3
    wrong_class = (labels + 1) % c
    base[flip] = 0.0
    base[flip.nonzero().squeeze(), wrong_class[flip]] = 1.0
    return base * scale, labels


class TestTemperatureScaling:
    def test_apply_temperature_preserves_argmax(self):
        logits = torch.tensor([[1.0, 2.0, 0.5]])
        scaled = apply_temperature(logits, temperature=3.0)
        assert scaled.argmax(dim=1).item() == logits.argmax(dim=1).item()

    def test_apply_temperature_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            apply_temperature(torch.zeros(1, 2), temperature=0.0)

    def test_fit_temperature_reduces_ece(self):
        logits, labels = _make_overconfident_logits()
        probs_before = torch.softmax(logits, dim=1)
        ece_before = compute_ece(probs_before, labels)

        t = fit_temperature(logits, labels, max_iter=50)
        assert t > 1.0, f"expected temperature > 1 for over-confident model, got {t}"

        probs_after = torch.softmax(apply_temperature(logits, t), dim=1)
        ece_after = compute_ece(probs_after, labels)
        assert ece_after < ece_before

    def test_fit_temperature_validates_shapes(self):
        with pytest.raises(ValueError):
            fit_temperature(torch.zeros(4), torch.zeros(4, dtype=torch.long))
        with pytest.raises(ValueError):
            fit_temperature(torch.zeros(4, 2), torch.zeros(3, dtype=torch.long))


class TestECE:
    def test_perfect_calibration_has_zero_ece(self):
        # 10 samples at confidence ≈ 1.0, all correct → bin acc = bin conf → ECE = 0.
        probs = torch.tensor([[1.0 - 1e-6, 1e-6]] * 10 + [[1e-6, 1.0 - 1e-6]] * 10)
        labels = torch.tensor([0] * 10 + [1] * 10)
        assert compute_ece(probs, labels) < 1e-3

    def test_overconfident_has_positive_ece(self):
        logits, labels = _make_overconfident_logits(scale=10.0)
        probs = torch.softmax(logits, dim=1)
        assert compute_ece(probs, labels) > 0.1


class TestReliabilityCurve:
    def test_bins_sum_to_total(self):
        probs = torch.rand(200, 3)
        probs = probs / probs.sum(dim=1, keepdim=True)
        labels = torch.randint(0, 3, (200,))
        bins = reliability_curve(probs, labels, n_bins=10)
        assert sum(b.count for b in bins) == 200
