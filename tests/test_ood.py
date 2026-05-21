"""Tests for the Gaussian OOD detector."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from medqcnn.inference.ood import OODDetector


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


def _in_dist(n: int = 500, d: int = 16) -> np.ndarray:
    return np.random.randn(n, d).astype(np.float32)


def _out_of_dist(n: int = 200, d: int = 16, shift: float = 8.0) -> np.ndarray:
    return np.random.randn(n, d).astype(np.float32) + shift


class TestFit:
    def test_rejects_non_2d_input(self):
        det = OODDetector()
        with pytest.raises(ValueError):
            det.fit(np.zeros(4))

    def test_sets_feature_dim_and_n_train(self):
        det = OODDetector()
        det.fit(_in_dist(n=300, d=10))
        assert det.is_fitted
        assert det.feature_dim == 10
        assert det.n_train == 300


class TestScore:
    def test_score_before_fit_raises(self):
        det = OODDetector()
        with pytest.raises(RuntimeError):
            det.score(np.zeros((1, 4)))

    def test_in_dist_scores_lower_than_ood(self):
        det = OODDetector()
        det.fit(_in_dist())
        in_nll = det.score(_in_dist(n=100))
        ood_nll = det.score(_out_of_dist(n=100))
        assert ood_nll.mean() > in_nll.mean()

    def test_feature_dim_mismatch_raises(self):
        det = OODDetector()
        det.fit(_in_dist(d=16))
        with pytest.raises(ValueError):
            det.score(np.zeros((1, 8)))

    def test_accepts_torch_tensors(self):
        det = OODDetector()
        det.fit(torch.from_numpy(_in_dist()))
        s = det.score(torch.from_numpy(_in_dist(n=10)))
        assert s.shape == (10,)


class TestThresholdAndIsOod:
    def test_is_ood_before_threshold_raises(self):
        det = OODDetector()
        det.fit(_in_dist())
        with pytest.raises(RuntimeError):
            det.is_ood(_in_dist(n=1))

    def test_threshold_separates_classes(self):
        det = OODDetector()
        det.fit(_in_dist(n=1000))
        det.calibrate_threshold(_in_dist(n=500), percentile=95.0)

        ood_flags = det.is_ood(_out_of_dist(n=200, shift=10.0))
        # At shift=10 the OOD samples should be flagged at very high rate
        assert sum(ood_flags) >= 180  # >= 90%

        in_flags = det.is_ood(_in_dist(n=500))
        # In-distribution false-positive rate at P95 ~ 5%
        assert sum(in_flags) <= 60


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        det = OODDetector()
        det.fit(_in_dist())
        det.calibrate_threshold(_in_dist(n=200))
        path = tmp_path / "ood.json"
        det.save(path)

        reloaded = OODDetector.load(path)
        assert reloaded.feature_dim == det.feature_dim
        assert reloaded.threshold == det.threshold
        assert reloaded.percentile == det.percentile

        # Scores should match within tolerance
        x = _in_dist(n=5)
        np.testing.assert_allclose(reloaded.score(x), det.score(x), rtol=1e-10)

    def test_save_before_fit_raises(self, tmp_path):
        det = OODDetector()
        with pytest.raises(RuntimeError):
            det.save(tmp_path / "ood.json")
