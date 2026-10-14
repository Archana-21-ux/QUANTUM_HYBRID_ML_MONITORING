import numpy as np
import pandas as pd
import pytest

from quantumguard.drift.injection import apply_drift, build_stream, drift_intensity

RNG = np.random.default_rng(3)


def make_pool(n=20000, fraud_rate=0.05):
    is_fraud = (RNG.random(n) < fraud_rate).astype(int)
    return pd.DataFrame(
        {
            "type": RNG.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], size=n),
            "step": RNG.integers(1, 700, size=n),
            "amount": RNG.exponential(1000, size=n),
            "oldbalanceOrg": RNG.exponential(5000, size=n),
            "newbalanceOrig": RNG.exponential(5000, size=n),
            "oldbalanceDest": RNG.exponential(5000, size=n),
            "newbalanceDest": RNG.exponential(5000, size=n),
            "isFraud": is_fraud,
        }
    )


def make_stream(pool, n_rows=2000, window_size=100):
    stream = build_stream(pool, n_rows=n_rows, fraud_rate=0.08, seed=1)
    stream.insert(0, "window_id", stream.index // window_size)
    return stream


DRIFT_KWARGS = dict(
    drift_start_window=10,
    ramp_windows=5,
    scale=2.0,
    offset_std=1.0,
    shift_features=["amount"],
)


def test_build_stream_size_and_fraud_rate():
    pool = make_pool()
    stream = build_stream(pool, n_rows=5000, fraud_rate=0.08, seed=1)
    assert len(stream) == 5000
    assert stream["isFraud"].mean() == pytest.approx(0.08, abs=0.001)
    # deterministic given seed
    again = build_stream(pool, n_rows=5000, fraud_rate=0.08, seed=1)
    pd.testing.assert_frame_equal(stream, again)


def test_drift_intensity_shapes():
    windows = np.arange(20)
    sudden = drift_intensity(windows, kind="sudden", drift_start_window=10, ramp_windows=5)
    assert np.all(sudden[:10] == 0.0) and np.all(sudden[10:] == 1.0)

    gradual = drift_intensity(windows, kind="gradual", drift_start_window=10, ramp_windows=5)
    assert np.all(gradual[:10] == 0.0)
    assert np.all(np.diff(gradual[10:15]) > 0), "ramp must increase"
    assert np.all(gradual[14:] == 1.0)

    none = drift_intensity(windows, kind="none", drift_start_window=10, ramp_windows=5)
    assert np.all(none == 0.0)


def test_apply_sudden_shifts_only_post_start_windows():
    pool = make_pool()
    stream = make_stream(pool)
    stds = {"amount": float(stream["amount"].std())}
    drifted = apply_drift(stream, kind="sudden", feature_stds=stds, **DRIFT_KWARGS)

    pre = stream["window_id"] < 10
    pd.testing.assert_series_equal(drifted.loc[pre, "amount"], stream.loc[pre, "amount"])
    assert (drifted.loc[~pre, "amount"] > stream.loc[~pre, "amount"]).all()
    # untouched features stay identical everywhere
    pd.testing.assert_series_equal(drifted["oldbalanceOrg"], stream["oldbalanceOrg"])


def test_apply_fraud_subset_shifts_only_fraud_rows():
    pool = make_pool()
    stream = make_stream(pool)
    stds = {"amount": float(stream["amount"].std())}
    drifted = apply_drift(stream, kind="fraud_subset", feature_stds=stds, **DRIFT_KWARGS)

    post = stream["window_id"] >= 10
    fraud = stream["isFraud"] == 1
    unchanged = ~(post & fraud)
    pd.testing.assert_series_equal(drifted.loc[unchanged, "amount"], stream.loc[unchanged, "amount"])
    assert (drifted.loc[post & fraud, "amount"] > stream.loc[post & fraud, "amount"]).all()


def test_apply_gradual_mixes_regimes_with_ramping_fraction():
    pool = make_pool()
    stream = make_stream(pool)
    stds = {"amount": float(stream["amount"].std())}
    drifted = apply_drift(stream, kind="gradual", feature_stds=stds, seed=9, **DRIFT_KWARGS)

    changed = (drifted["amount"] != stream["amount"]).groupby(stream["window_id"]).mean()
    assert (changed.loc[:9] == 0.0).all(), "no rows may change before drift start"
    assert changed.loc[15:].min() > 0.9, "post-ramp windows must be (almost) fully shifted"
    ramp = changed.loc[10:14]
    assert ramp.is_monotonic_increasing or (ramp.diff().dropna() > -0.15).all(), (
        "changed-row fraction must ramp upward"
    )
    # changed rows take the FULL shift, not a partial one
    mask = drifted["amount"] != stream["amount"]
    np.testing.assert_allclose(
        drifted.loc[mask, "amount"],
        stream.loc[mask, "amount"] * 2.0 + stds["amount"],
        rtol=1e-9,
    )


def test_apply_none_is_identity():
    pool = make_pool()
    stream = make_stream(pool)
    stds = {"amount": float(stream["amount"].std())}
    drifted = apply_drift(stream, kind="none", feature_stds=stds, **DRIFT_KWARGS)
    pd.testing.assert_frame_equal(drifted, stream)
