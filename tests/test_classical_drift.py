import numpy as np
import pytest

from quantumguard.config import load_config
from quantumguard.drift.base import DriftDetector
from quantumguard.drift.classical import ClassicalDriftDetector, classical_drift_scores

RNG = np.random.default_rng(42)
N_REF, N_CUR, N_FEATURES = 2000, 500, 4

STAT_KWARGS = dict(n_bins=10, smoothing=1e-4, psi_cap=0.5, kl_cap=1.0)


@pytest.fixture(scope="module")
def reference():
    return RNG.normal(0.0, 1.0, size=(N_REF, N_FEATURES))


@pytest.fixture(scope="module")
def current_same():
    return RNG.normal(0.0, 1.0, size=(N_CUR, N_FEATURES))


@pytest.fixture(scope="module")
def current_shifted():
    return RNG.normal(1.0, 1.0, size=(N_CUR, N_FEATURES))


def test_no_shift_scores_near_zero(reference, current_same):
    result = classical_drift_scores(reference, current_same, **STAT_KWARGS)
    for stat in ("psi", "kl", "ks", "hellinger", "combined"):
        assert getattr(result, stat) < 0.15, f"{stat} too high on identical distributions"
    assert not result.low_confidence


def test_every_stat_rises_on_shift(reference, current_same, current_shifted):
    base = classical_drift_scores(reference, current_same, **STAT_KWARGS)
    drifted = classical_drift_scores(reference, current_shifted, **STAT_KWARGS)
    for stat in ("psi", "kl", "ks", "hellinger", "combined"):
        assert getattr(drifted, stat) > getattr(base, stat) + 0.2, f"{stat} did not rise on shift"


def test_scores_bounded(reference):
    extreme = RNG.normal(50.0, 1.0, size=(N_CUR, N_FEATURES))
    result = classical_drift_scores(reference, extreme, **STAT_KWARGS)
    for stat in ("psi", "kl", "ks", "hellinger", "combined"):
        value = getattr(result, stat)
        assert 0.0 <= value <= 1.0, f"{stat}={value} out of [0, 1]"
    assert result.combined > 0.9


def test_kl_stable_on_sparse_fraud_subset(reference):
    """Risk item: KL on tiny fraud-subset windows (n < 30) must stay finite."""
    tiny_ref = RNG.normal(0.0, 1.0, size=(25, N_FEATURES))
    tiny_cur = RNG.normal(0.0, 1.0, size=(18, N_FEATURES))
    result = classical_drift_scores(tiny_ref, tiny_cur, **STAT_KWARGS)
    for stat in ("psi", "kl", "ks", "hellinger", "combined"):
        value = getattr(result, stat)
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_subset_masks_isolate_shifted_subpopulation(reference, current_same):
    """A shift confined to a masked subset must score higher on the subset
    than on the diluted full population."""
    subset_frac = 0.1
    cur = current_same.copy()
    mask_cur = RNG.random(len(cur)) < subset_frac
    cur[mask_cur] += 3.0
    mask_ref = RNG.random(len(reference)) < subset_frac

    population = classical_drift_scores(reference, cur, **STAT_KWARGS)
    subset = classical_drift_scores(
        reference, cur, reference_mask=mask_ref, current_mask=mask_cur, **STAT_KWARGS
    )
    assert subset.combined > population.combined + 0.2
    assert subset.n_current == int(mask_cur.sum())


def test_below_min_subset_n_returns_low_confidence(reference):
    tiny = RNG.normal(0.0, 1.0, size=(5, N_FEATURES))
    result = classical_drift_scores(reference, tiny, min_subset_n=30, **STAT_KWARGS)
    assert result.low_confidence
    assert result.combined == 0.0


def test_detector_implements_protocol(reference, current_same, current_shifted):
    detector = ClassicalDriftDetector(load_config())
    assert isinstance(detector, DriftDetector)
    assert detector.score(reference, current_same) < 0.15
    assert detector.score(reference, current_shifted) > 0.3
