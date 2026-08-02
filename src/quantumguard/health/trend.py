from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import linregress

from quantumguard.config import Config, load_config


@dataclass(frozen=True)
class TrendDecision:
    """Complete audit record of one trend evaluation. Every input that shaped
    the decision is captured so the reasoning can be replayed exactly."""

    window_id: int
    series: tuple[float, ...]
    slope: float
    p_value: float
    cusum_stat: float
    cusum_threshold: float
    baseline_mean: float
    baseline_std: float
    triggered: bool
    trigger_reason: str | None   # "slope" | "cusum" | "slope+cusum" | None
    not_triggered_reason: str | None  # "insufficient_history" | "mhs_healthy" | "no_signal" | None
    drift_type: str | None       # "sudden" | "gradual" (triggered decisions only)
    retrain_window_width: int | None


def evaluate_trend(
    mhs_series: Sequence[float],
    config: Config | None = None,
    *,
    window_id: int | None = None,
) -> TrendDecision:
    """Pure, deterministic retrain-trigger decision from an MHS series.

    Same series in -> same decision out. No I/O, no randomness, no LLM calls
    (CLAUDE.md: narration is downstream-only). Two detectors run on the series:

    - rolling slope: least-squares slope over the last `rolling_slope_window`
      points; fires when significantly negative (p < alpha).
    - CUSUM: one-sided cumulative sum of drops below the baseline mean (first
      `baseline_window` points), with slack and threshold in baseline-std units.

    Either can fire, but the `mhs_gate` keeps both quiet while current MHS is
    still in the healthy band. Triggered decisions classify the drift as
    sudden (recent sharp drop) or gradual, choosing the retrain window width.
    """
    cfg = config if config is not None else load_config()
    td = cfg.trend_detector
    series = tuple(float(x) for x in mhs_series)
    wid = window_id if window_id is not None else len(series) - 1

    def decision(**overrides) -> TrendDecision:
        fields = dict(
            window_id=wid,
            series=series,
            slope=0.0,
            p_value=1.0,
            cusum_stat=0.0,
            cusum_threshold=float(td.cusum_threshold),
            baseline_mean=0.0,
            baseline_std=0.0,
            triggered=False,
            trigger_reason=None,
            not_triggered_reason=None,
            drift_type=None,
            retrain_window_width=None,
        )
        fields.update(overrides)
        return TrendDecision(**fields)

    if len(series) < td.min_series_length:
        return decision(not_triggered_reason="insufficient_history")

    # rolling slope over the trailing window
    tail = np.array(series[-td.rolling_slope_window :])
    fit = linregress(np.arange(len(tail)), tail)
    slope, p_value = float(fit.slope), float(fit.pvalue)

    # one-sided CUSUM of drops below the baseline, in baseline-std units
    baseline = np.array(series[: td.baseline_window])
    baseline_mean = float(baseline.mean())
    baseline_std = max(float(baseline.std()), td.std_floor)
    slack = td.cusum_drift_slack * baseline_std
    cusum = 0.0
    for value in series[td.baseline_window :]:
        cusum = max(0.0, cusum + (baseline_mean - value) - slack)
    cusum_stat = cusum / baseline_std
    threshold = float(td.cusum_threshold)

    common = dict(
        slope=slope,
        p_value=p_value,
        cusum_stat=cusum_stat,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
    )

    if series[-1] >= td.mhs_gate:
        return decision(not_triggered_reason="mhs_healthy", **common)

    slope_fired = slope < 0.0 and p_value < td.slope_significance_alpha
    cusum_fired = cusum_stat > threshold
    if not (slope_fired or cusum_fired):
        return decision(not_triggered_reason="no_signal", **common)

    reason = "+".join(
        name for name, fired in (("slope", slope_fired), ("cusum", cusum_fired)) if fired
    )

    lookback = min(td.sudden_drop_lookback, len(series) - 1)
    recent_drop = series[-1 - lookback] - series[-1]
    drift_type = "sudden" if recent_drop >= td.sudden_drop_threshold else "gradual"

    return decision(
        triggered=True,
        trigger_reason=reason,
        drift_type=drift_type,
        retrain_window_width=int(td.retrain_window_width[drift_type]),
        **common,
    )
