import numpy as np
import pandas as pd
import pytest

from quantumguard.config import load_config
from quantumguard.drift.feature_breakdown import FeatureDriftAnalyzer

RNG = np.random.default_rng(77)


def make_frame(n=2000, amount_shift=0.0, type_mix=None):
    """Synthetic PaySim-shaped frame. type_mix overrides category probabilities."""
    mix = type_mix or {"PAYMENT": 0.5, "TRANSFER": 0.25, "CASH_OUT": 0.25}
    return pd.DataFrame(
        {
            "type": RNG.choice(list(mix), size=n, p=list(mix.values())),
            "step": RNG.integers(1, 700, size=n),
            "amount": RNG.exponential(1000, size=n) + amount_shift,
            "oldbalanceOrg": RNG.exponential(5000, size=n),
            "newbalanceOrig": RNG.exponential(5000, size=n),
            "oldbalanceDest": RNG.exponential(5000, size=n),
            "newbalanceDest": RNG.exponential(5000, size=n),
            "isFraud": np.zeros(n, dtype=int),
        }
    )


@pytest.fixture(scope="module")
def analyzer():
    return FeatureDriftAnalyzer(load_config()).fit_reference(make_frame(4000))


def test_tracks_the_seven_paysim_features(analyzer):
    rows = analyzer.analyze(make_frame(500))
    assert {row.feature for row in rows} == {
        "Amount", "Transaction Type", "Origin Old Balance", "Origin New Balance",
        "Destination Old Balance", "Destination New Balance", "Hour",
    }
    # nameDest (near-unique account ID) must NOT be tracked — PSI/KL/KS on a
    # high-cardinality identifier is not statistically meaningful
    assert not any("Account" in row.feature for row in rows)


def test_no_drift_all_features_near_zero(analyzer):
    rows = analyzer.analyze(make_frame(500))
    assert len(rows) == 7
    for row in rows:
        assert 0.0 <= row.classical_drift <= 1.0
        assert 0.0 <= row.quantum_drift <= 1.0
        assert row.hybrid_drift < 0.2, f"{row.feature} drifted on identical data"
        assert row.severity == "ok"


def test_amount_shift_ranks_amount_first(analyzer):
    rows = analyzer.analyze(make_frame(500, amount_shift=5000.0))
    assert rows[0].feature == "Amount"
    assert rows[0].classical_drift > 0.4
    assert rows[0].quantum_drift > 0.2
    # untouched features stay quiet
    others = {row.feature: row.hybrid_drift for row in rows[1:]}
    assert all(value < 0.2 for value in others.values()), others


def test_type_mix_shift_ranks_type_first(analyzer):
    rows = analyzer.analyze(
        make_frame(500, type_mix={"PAYMENT": 0.05, "TRANSFER": 0.05, "CASH_OUT": 0.9})
    )
    assert rows[0].feature == "Transaction Type"
    assert rows[0].hybrid_drift > 0.2


def test_hybrid_is_mean_and_sorted(analyzer):
    rows = analyzer.analyze(make_frame(500, amount_shift=2000.0))
    for row in rows:
        assert row.hybrid_drift == pytest.approx(
            (row.classical_drift + row.quantum_drift) / 2, abs=1e-3
        )
    hybrids = [row.hybrid_drift for row in rows]
    assert hybrids == sorted(hybrids, reverse=True)


def test_severity_bands(analyzer):
    assert analyzer._severity(0.1) == "ok"
    assert analyzer._severity(0.3) == "warn"
    assert analyzer._severity(0.7) == "crit"


def test_hour_of_day_derivation(analyzer):
    # a window whose steps all fall in night hours must drift on Hour only
    frame = make_frame(500)
    frame["step"] = RNG.integers(0, 3, size=len(frame)) + 24 * RNG.integers(1, 20, size=len(frame))
    rows = analyzer.analyze(frame)
    assert rows[0].feature == "Hour"
    assert rows[0].hybrid_drift > 0.3


def test_analyze_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit_reference"):
        FeatureDriftAnalyzer(load_config()).analyze(make_frame(100))
