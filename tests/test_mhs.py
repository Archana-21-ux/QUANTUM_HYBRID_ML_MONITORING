import copy

import numpy as np
import pandas as pd
import pytest

from quantumguard.config import Config, load_config
from quantumguard.health.mhs import ModelHealthScorer, status_for

RNG = np.random.default_rng(11)
DRIFT_COLUMNS = ["f1", "f2"]
WINDOW_SIZE = 200
FRAUD_RATE = 0.2  # 40 fraud rows per window, above min_fraud_subset_n


class LabelReadingModel:
    """Test stub: predicts from the label column itself. accuracy=1.0 when
    inverted=False; a perfectly wrong model when inverted=True."""

    def __init__(self, inverted: bool = False):
        self.inverted = inverted

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = np.where(X["isFraud"].to_numpy() == 1, 0.95, 0.05)
        if self.inverted:
            p1 = 1.0 - p1
        return np.column_stack([1.0 - p1, p1])


def make_frame(n: int, shift: float = 0.0, fraud_shift: float = 0.0) -> pd.DataFrame:
    # deterministic fraud count (exactly n*FRAUD_RATE) so the min_fraud_subset_n
    # gate can never flake on an unlucky draw; feature noise stays random
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[: int(n * FRAUD_RATE)] = 1
    is_fraud = RNG.permutation(is_fraud)
    data = RNG.normal(shift, 1.0, size=(n, len(DRIFT_COLUMNS)))
    data[is_fraud == 1] += fraud_shift
    frame = pd.DataFrame(data, columns=DRIFT_COLUMNS)
    frame["isFraud"] = is_fraud
    return frame


def make_scorer(model=None, config=None) -> ModelHealthScorer:
    scorer = ModelHealthScorer(
        model if model is not None else LabelReadingModel(),
        feature_columns=["isFraud"],
        drift_columns=DRIFT_COLUMNS,
        config=config,
    )
    return scorer.fit_reference(make_frame(2000))


def test_weights_must_sum_to_one():
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["mhs"]["weights"]["cds"] = 0.5  # now sums to 1.15
    with pytest.raises(ValueError, match="sum to 1.0"):
        ModelHealthScorer(
            LabelReadingModel(), feature_columns=["isFraud"], drift_columns=DRIFT_COLUMNS, config=cfg
        )


def test_healthy_window_scores_healthy():
    scorer = make_scorer()
    components = scorer.score_window(make_frame(WINDOW_SIZE), window_id=0)
    assert components.status == "healthy"
    assert components.mhs > 0.85
    for field in ("cds", "qds", "pc", "at", "mhs"):
        assert 0.0 <= getattr(components, field) <= 1.0


def test_population_drift_degrades_mhs():
    scorer = make_scorer()
    healthy = scorer.score_window(make_frame(WINDOW_SIZE), window_id=0)
    drifted = scorer.score_window(make_frame(WINDOW_SIZE, shift=2.0), window_id=1)
    assert drifted.cds > healthy.cds + 0.3
    assert drifted.mhs < healthy.mhs - 0.15


def test_fraud_subset_drift_fused_via_max():
    scorer = make_scorer()
    components = scorer.score_window(
        make_frame(WINDOW_SIZE, shift=0.0, fraud_shift=3.0), window_id=0
    )
    assert components.cds_fraud is not None
    assert components.cds_fraud > components.cds_population + 0.2
    assert components.cds == pytest.approx(
        max(components.cds_population, components.cds_fraud)
    )


def test_label_latency_delays_accuracy_trend():
    """A perfectly wrong model must look healthy on AT until the first lagged
    labels arrive, then crash."""
    cfg = load_config()
    lag = cfg.mhs.label_latency_windows
    scorer = make_scorer(model=LabelReadingModel(inverted=True))

    at_series = [
        scorer.score_window(make_frame(WINDOW_SIZE), window_id=i).at for i in range(lag + 1)
    ]
    assert all(at == 1.0 for at in at_series[:lag]), "AT must stay neutral before labels arrive"
    assert at_series[lag] < 0.1, "once window 0's labels arrive, the wrong model must be exposed"


def test_status_bands():
    bands = load_config().mhs.status_bands
    assert status_for(0.9, bands) == "healthy"
    assert status_for(0.75, bands) == "healthy"
    assert status_for(0.6, bands) == "warning"
    assert status_for(0.5, bands) == "warning"
    assert status_for(0.49, bands) == "critical"


def test_score_before_fit_raises():
    scorer = ModelHealthScorer(
        LabelReadingModel(), feature_columns=["isFraud"], drift_columns=DRIFT_COLUMNS
    )
    with pytest.raises(RuntimeError, match="fit_reference"):
        scorer.score_window(make_frame(WINDOW_SIZE), window_id=0)
