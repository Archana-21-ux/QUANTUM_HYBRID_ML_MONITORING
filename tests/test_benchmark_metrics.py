import copy

import numpy as np
import pytest

from quantumguard.benchmark.metrics import (
    assert_benchmark_split,
    calibrate_threshold,
    evaluate_series,
)
from quantumguard.config import Config, load_config


def test_split_assertion_passes_on_default_config():
    held_out, evaluation = assert_benchmark_split()
    assert held_out and evaluation
    assert not set(held_out) & set(evaluation)


def test_split_assertion_catches_leak():
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["benchmark"]["evaluation_scenarios"].append(cfg["benchmark"]["held_out_scenarios"][0])
    with pytest.raises(ValueError, match="leaks"):
        assert_benchmark_split(cfg)


def test_calibrate_threshold():
    scores = np.array([0.1, 0.12, 0.08, 0.1, np.nan])
    threshold = calibrate_threshold(scores, sigmas=3.0)
    clean = scores[:4]
    assert threshold == pytest.approx(clean.mean() + 3.0 * clean.std())
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([0.1]), sigmas=3.0)


def test_detection_delay_and_persistence():
    scores = np.array([0.1] * 10 + [0.9, 0.1, 0.9, 0.9, 0.9])
    # persistence 1: the lone spike at window 10 counts -> delay 0
    r1 = evaluate_series(scores, threshold=0.5, drift_start=10, persistence=1)
    assert r1.detected and r1.delay == 0
    # persistence 2: first pair of consecutive alarms starts at window 12 -> delay 2
    r2 = evaluate_series(scores, threshold=0.5, drift_start=10, persistence=2)
    assert r2.detected and r2.delay == 2


def test_missed_detection():
    scores = np.array([0.1] * 20)
    result = evaluate_series(scores, threshold=0.5, drift_start=10, persistence=2)
    assert not result.detected
    assert result.delay is None
    assert result.false_alarm_rate == 0.0


def test_false_alarms_counted_pre_drift_only():
    scores = np.array([0.9, 0.9] + [0.1] * 8 + [0.9] * 10)
    result = evaluate_series(scores, threshold=0.5, drift_start=10, persistence=2)
    assert result.false_alarm_rate == pytest.approx(0.1)  # one persistent pre-drift alarm at w0
    assert result.delay == 0


def test_control_scenario_uses_all_windows():
    scores = np.array([0.1] * 18 + [0.9, 0.9])
    result = evaluate_series(scores, threshold=0.5, drift_start=None, persistence=2)
    assert result.delay is None and result.separation is None
    assert result.false_alarm_rate == pytest.approx(1 / 20)


def test_nan_windows_are_skipped():
    scores = np.array([0.1, np.nan, 0.1, 0.1] * 3 + [0.9] * 4)
    result = evaluate_series(scores, threshold=0.5, drift_start=12, persistence=2)
    assert result.detected and result.delay == 0
    assert result.false_alarm_rate == 0.0


def test_separation_index():
    scores = np.array([0.1] * 10 + [0.7] * 10)
    result = evaluate_series(scores, threshold=0.5, drift_start=10, persistence=1)
    assert result.separation > 100  # pre std is ~0 -> huge separation
