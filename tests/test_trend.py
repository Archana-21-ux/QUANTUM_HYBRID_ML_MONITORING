"""Golden regression tests: the frozen fixtures in tests/fixtures/mhs_series
are the behavioral contract of the trend detector. Regenerate them only via
the _generate.py script when intended behavior changes."""

import json
from pathlib import Path

import pytest

from quantumguard.config import load_config
from quantumguard.health.trend import evaluate_trend

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "mhs_series"
FIXTURE_PATHS = sorted(FIXTURES_DIR.glob("*.json"))

FLOAT_FIELDS = ("slope", "p_value", "cusum_stat")


@pytest.mark.parametrize("fixture_path", FIXTURE_PATHS, ids=lambda p: p.stem)
def test_golden_series(fixture_path):
    with open(fixture_path) as f:
        fixture = json.load(f)

    decision = evaluate_trend(fixture["series"])
    for field, expected in fixture["expected"].items():
        actual = getattr(decision, field)
        if field in FLOAT_FIELDS:
            assert actual == pytest.approx(expected, rel=1e-6), field
        else:
            assert actual == expected, field


def test_fixture_battery_is_complete():
    names = {p.stem for p in FIXTURE_PATHS}
    assert {"healthy_flat", "steady_decline", "sudden_drop", "noisy_blip", "flat_warning"} <= names


def test_deterministic_and_pure():
    series = [0.9] * 20 + [0.6, 0.5, 0.45]
    original = list(series)
    first = evaluate_trend(series)
    second = evaluate_trend(series)
    assert first == second
    assert series == original, "evaluate_trend must not mutate its input"


def test_insufficient_history():
    decision = evaluate_trend([0.4, 0.3])
    assert not decision.triggered
    assert decision.not_triggered_reason == "insufficient_history"


def test_decision_records_full_series():
    series = [0.9] * 12
    decision = evaluate_trend(series)
    assert decision.series == tuple(series)
    assert decision.window_id == len(series) - 1


def test_healthy_gate_blocks_trigger():
    # significant decline but still inside the healthy band -> gated
    cfg = load_config()
    gate = cfg.trend_detector.mhs_gate
    series = [0.98 - 0.005 * i for i in range(20)]  # ends at 0.885 > gate
    assert series[-1] > gate
    decision = evaluate_trend(series)
    assert not decision.triggered
    assert decision.not_triggered_reason == "mhs_healthy"
