"""Pure metric functions for the Phase 4 detector benchmark.

All functions operate on plain per-window score arrays (NaN = window skipped,
e.g. a fraud subset below the minimum sample gate) so they are trivially
testable and independent of how the scores were produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantumguard.config import Config, load_config


def assert_benchmark_split(config: Config | None = None) -> tuple[list[str], list[str]]:
    """Leakage guard (risk item #3): the held-out and evaluation scenario sets
    must be disjoint and non-empty. Raises instead of warning — a leaky split
    silently invalidates every downstream number."""
    cfg = config if config is not None else load_config()
    held_out = list(cfg.benchmark.held_out_scenarios)
    evaluation = list(cfg.benchmark.evaluation_scenarios)
    if not held_out or not evaluation:
        raise ValueError("benchmark split: both scenario sets must be non-empty")
    overlap = set(held_out) & set(evaluation)
    if overlap:
        raise ValueError(
            f"benchmark split leaks: {sorted(overlap)} appear in BOTH held-out and evaluation sets"
        )
    return held_out, evaluation


def calibrate_threshold(
    pre_drift_scores: np.ndarray, *, sigmas: float, minimum: float = 0.0
) -> float:
    """Detection threshold from drift-free calibration scores: mean + sigmas*std."""
    scores = np.asarray(pre_drift_scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if len(scores) < 2:
        raise ValueError("need at least 2 calibration scores")
    return max(float(scores.mean() + sigmas * scores.std()), minimum)


def _alarm_mask(scores: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    """True at window w when scores[w : w+persistence] are all above threshold
    (NaN windows never count as above)."""
    above = np.asarray(scores, dtype=float) > threshold  # NaN > x is False
    if persistence <= 1:
        return above
    mask = np.zeros(len(above), dtype=bool)
    for start in range(len(above) - persistence + 1):
        if above[start : start + persistence].all():
            mask[start] = True
    return mask


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    delay: int | None            # windows from drift start to first persistent alarm
    false_alarm_rate: float      # fraction of eligible drift-free windows raising an alarm
    separation: float | None     # (post mean - pre mean) / pre std; None for controls
    threshold: float


def evaluate_series(
    scores: np.ndarray,
    *,
    threshold: float,
    drift_start: int | None,
    persistence: int,
) -> DetectionResult:
    """Detection metrics for one score series.

    drift_start=None marks a no-drift control: every window is eligible for
    false alarms and delay/separation are undefined.
    """
    scores = np.asarray(scores, dtype=float)
    alarms = _alarm_mask(scores, threshold, persistence)

    if drift_start is None:
        eligible = ~np.isnan(scores)
        far = float(alarms[eligible].mean()) if eligible.any() else 0.0
        return DetectionResult(
            detected=False, delay=None, false_alarm_rate=far, separation=None, threshold=threshold
        )

    pre, post = scores[:drift_start], scores[drift_start:]
    pre_eligible = ~np.isnan(pre)
    far = float(alarms[:drift_start][pre_eligible].mean()) if pre_eligible.any() else 0.0

    post_alarms = np.flatnonzero(alarms[drift_start:])
    detected = len(post_alarms) > 0
    delay = int(post_alarms[0]) if detected else None

    pre_clean, post_clean = pre[~np.isnan(pre)], post[~np.isnan(post)]
    separation = None
    if len(pre_clean) >= 2 and len(post_clean) >= 1:
        separation = float(
            (post_clean.mean() - pre_clean.mean()) / (pre_clean.std() + 1e-9)
        )

    return DetectionResult(
        detected=detected,
        delay=delay,
        false_alarm_rate=far,
        separation=separation,
        threshold=threshold,
    )
