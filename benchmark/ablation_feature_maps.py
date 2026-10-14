"""Phase 4 feature-map ablation, on HELD-OUT scenarios only.

Compares the three PennyLane feature maps (angle = no-entanglement control,
ZZ/IQP, data re-uploading) on detection delay, false alarms, separation, and
wall-clock, using the same cached score series as run_benchmark.py. The
selected map is frozen into configs/default.yaml (quantum_kernel.feature_map)
BEFORE the evaluation scenarios are ever consulted.

Usage: uv run python benchmark/ablation_feature_maps.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantumguard.benchmark.metrics import (
    assert_benchmark_split,
    calibrate_threshold,
    evaluate_series,
)
from quantumguard.config import load_config
from quantumguard.drift.injection import load_scenario
from quantumguard.drift.quantum_kernel import FEATURE_MAPS

BENCH_DIR = Path(__file__).parent
SERIES_PATH = BENCH_DIR / "score_series.csv"
ABLATION_PATH = BENCH_DIR / "ablation_results.csv"

QUANTUM_DETECTORS = [f"quantum_{m}" for m in FEATURE_MAPS]


def run_ablation() -> pd.DataFrame:
    cfg = load_config()
    held_out, _ = assert_benchmark_split(cfg)
    if not SERIES_PATH.exists():
        raise SystemExit("run benchmark/run_benchmark.py first to build score_series.csv")
    series = pd.read_csv(SERIES_PATH)

    rows = []
    for scenario in held_out:
        meta = load_scenario(scenario, cfg)[2]
        drift_start = meta["drift_start_window"]
        for detector in QUANTUM_DETECTORS:
            for subset in ("population", "fraud"):
                group = series[
                    (series["scenario"] == scenario)
                    & (series["detector"] == detector)
                    & (series["subset"] == subset)
                ].sort_values("window_id")
                scores = group["score"].to_numpy()
                threshold = calibrate_threshold(
                    scores[:drift_start], sigmas=cfg.benchmark.threshold_sigmas
                )
                result = evaluate_series(
                    scores,
                    threshold=threshold,
                    drift_start=drift_start,
                    persistence=cfg.benchmark.detection_persistence,
                )
                rows.append(
                    {
                        "feature_map": detector.removeprefix("quantum_"),
                        "scenario": scenario,
                        "subset": subset,
                        "detected": result.detected,
                        "delay": result.delay,
                        "false_alarm_rate": round(result.false_alarm_rate, 4),
                        "separation": round(result.separation, 2),
                        "mean_wall_clock_ms": round(float(group["wall_clock_ms"].mean()), 1),
                    }
                )
    return pd.DataFrame(rows)


def select_map(ablation: pd.DataFrame) -> str:
    """Selection rule, in priority order: every scenario detected -> lowest mean
    delay -> lowest false alarms -> highest separation -> lowest wall-clock."""
    summary = (
        ablation.groupby("feature_map")
        .agg(
            all_detected=("detected", "all"),
            mean_delay=("delay", lambda d: np.nan if d.isna().any() else d.mean()),
            far=("false_alarm_rate", "mean"),
            separation=("separation", "mean"),
            wall_clock=("mean_wall_clock_ms", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(
        by=["all_detected", "mean_delay", "far", "separation", "wall_clock"],
        ascending=[False, True, True, False, True],
    )
    print("\nper-map summary (held-out only):")
    print(summary.round(2).to_string(index=False))
    return summary.iloc[0]["feature_map"]


if __name__ == "__main__":
    ablation = run_ablation()
    ablation.to_csv(ABLATION_PATH, index=False)
    print(ablation.to_string(index=False))

    selected = select_map(ablation)
    deployed = load_config().quantum_kernel.feature_map
    print(f"\nselected feature map: {selected} (config currently deploys: {deployed})")
    if selected != deployed:
        print("-> update quantum_kernel.feature_map in configs/default.yaml before final runs")
    print(f"wrote {ABLATION_PATH}")
