"""Regenerate the golden MHS-series fixtures.

Run only when the trend detector's intended behavior changes; the frozen
`expected` blocks are the regression contract.

Usage: uv run python tests/fixtures/mhs_series/_generate.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quantumguard.health.trend import evaluate_trend

FIXTURES_DIR = Path(__file__).parent

EXPECTED_FIELDS = (
    "triggered",
    "trigger_reason",
    "not_triggered_reason",
    "drift_type",
    "retrain_window_width",
    "slope",
    "p_value",
    "cusum_stat",
)


def build_series() -> dict[str, list[float]]:
    rng = np.random.default_rng(2026)
    noise = lambda n, s: rng.normal(0.0, s, n)  # noqa: E731

    healthy_flat = 0.90 + noise(30, 0.01)

    steady_decline = 0.95 - 0.012 * np.arange(40) + noise(40, 0.008)

    sudden_drop = np.concatenate([0.92 + noise(25, 0.008), [0.55, 0.42, 0.40]])

    blip = 0.85 + noise(30, 0.015)
    blip[15] = 0.65  # one bad window, fully recovered afterwards
    blip[-1] = max(blip[-1], 0.84)

    flat_warning = 0.70 + noise(30, 0.008)  # below the gate but not declining

    return {
        "healthy_flat": healthy_flat.round(4).tolist(),
        "steady_decline": steady_decline.round(4).tolist(),
        "sudden_drop": sudden_drop.round(4).tolist(),
        "noisy_blip": blip.round(4).tolist(),
        "flat_warning": flat_warning.round(4).tolist(),
    }


if __name__ == "__main__":
    for name, series in build_series().items():
        decision = asdict(evaluate_trend(series))
        expected = {field: decision[field] for field in EXPECTED_FIELDS}
        fixture = {"name": name, "series": series, "expected": expected}
        path = FIXTURES_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(fixture, f, indent=2)
        print(f"{name}: {expected}")
