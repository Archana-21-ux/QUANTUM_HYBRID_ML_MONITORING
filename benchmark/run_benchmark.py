"""Phase 4 detector benchmark: every DriftDetector x every drift scenario,
population AND fraud-subset views.

Protocol (leakage-guarded, risk item #3):
1. assert the held-out / evaluation split from config is disjoint;
2. compute per-window score series for all detectors on all scenarios
   (cached in benchmark/score_series.csv; --recompute to refresh);
3. calibrate per-detector thresholds ONLY from pre-drift windows of the
   held-out scenarios (frozen to benchmark/thresholds.json);
4. evaluate detection delay / false alarm rate / separation on the disjoint
   evaluation scenarios; write results.csv, per-scenario plots, and REPORT.md.

Usage: uv run python benchmark/run_benchmark.py [--recompute]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantumguard.benchmark.metrics import (
    assert_benchmark_split,
    calibrate_threshold,
    evaluate_series,
)
from quantumguard.config import load_config
from quantumguard.drift.baselines import DomainClassifierDetector, MMDRBFDetector
from quantumguard.drift.classical import ClassicalDriftDetector
from quantumguard.drift.injection import NUMERIC_FEATURES, load_scenario
from quantumguard.drift.quantum_kernel import QuantumKernelDriftDetector

BENCH_DIR = Path(__file__).parent
SERIES_PATH = BENCH_DIR / "score_series.csv"
THRESHOLDS_PATH = BENCH_DIR / "thresholds.json"
RESULTS_PATH = BENCH_DIR / "results.csv"
REPORT_PATH = BENCH_DIR / "REPORT.md"
INTERPRETATION_PATH = BENCH_DIR / "INTERPRETATION.md"
PLOTS_DIR = BENCH_DIR / "plots"


def build_detectors(cfg) -> list:
    return [
        ClassicalDriftDetector(cfg),
        QuantumKernelDriftDetector("angle", cfg),
        QuantumKernelDriftDetector("zz", cfg),
        QuantumKernelDriftDetector("reupload", cfg),
        MMDRBFDetector(cfg),
        DomainClassifierDetector(cfg),
    ]


def compute_score_series(cfg, scenario_names: list[str]) -> pd.DataFrame:
    """Long-format table: scenario, detector, subset, window_id, score, wall_clock_ms."""
    min_fraud_n = cfg.classical_drift.min_fraud_subset_n
    rows = []
    for scenario in scenario_names:
        reference, stream, _ = load_scenario(scenario, cfg)
        ref_X = reference[NUMERIC_FEATURES].to_numpy(dtype=float)
        ref_fraud = ref_X[(reference["isFraud"] == 1).to_numpy()]
        detectors = build_detectors(cfg)  # fresh per scenario: no cross-scenario state

        for window_id, window in stream.groupby("window_id"):
            cur_X = window[NUMERIC_FEATURES].to_numpy(dtype=float)
            cur_fraud = cur_X[(window["isFraud"] == 1).to_numpy()]
            fraud_ok = len(cur_fraud) >= min_fraud_n and len(ref_fraud) >= min_fraud_n

            for detector in detectors:
                for subset, ref_arr, cur_arr in (
                    ("population", ref_X, cur_X),
                    ("fraud", ref_fraud, cur_fraud),
                ):
                    if subset == "fraud" and not fraud_ok:
                        score, elapsed = np.nan, np.nan
                    else:
                        t0 = time.perf_counter()
                        score = detector.score(ref_arr, cur_arr)
                        elapsed = (time.perf_counter() - t0) * 1000.0
                    rows.append(
                        {
                            "scenario": scenario,
                            "detector": detector.name,
                            "subset": subset,
                            "window_id": int(window_id),
                            "score": score,
                            "wall_clock_ms": elapsed,
                        }
                    )
        print(f"scored {scenario}")
    return pd.DataFrame(rows)


def calibrate(series: pd.DataFrame, cfg, held_out: list[str]) -> dict:
    """Thresholds from held-out PRE-DRIFT windows only, per detector x subset."""
    drift_starts = {name: load_scenario(name, cfg)[2]["drift_start_window"] for name in held_out}
    calib = series[series["scenario"].isin(held_out)].copy()
    calib = calib[calib.apply(lambda r: r["window_id"] < drift_starts[r["scenario"]], axis=1)]

    thresholds: dict[str, dict[str, float]] = {}
    for (detector, subset), group in calib.groupby(["detector", "subset"]):
        thresholds.setdefault(detector, {})[subset] = calibrate_threshold(
            group["score"].to_numpy(), sigmas=cfg.benchmark.threshold_sigmas
        )
    return thresholds


def evaluate(series: pd.DataFrame, thresholds: dict, cfg, evaluation: list[str]) -> pd.DataFrame:
    persistence = cfg.benchmark.detection_persistence
    rows = []
    for scenario in evaluation:
        meta = load_scenario(scenario, cfg)[2]
        drift_start = None if meta["kind"] == "none" else meta["drift_start_window"]
        for (detector, subset), group in (
            series[series["scenario"] == scenario].groupby(["detector", "subset"])
        ):
            group = group.sort_values("window_id")
            result = evaluate_series(
                group["score"].to_numpy(),
                threshold=thresholds[detector][subset],
                drift_start=drift_start,
                persistence=persistence,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "kind": meta["kind"],
                    "detector": detector,
                    "subset": subset,
                    "detected": result.detected,
                    "delay": result.delay,
                    "false_alarm_rate": round(result.false_alarm_rate, 4),
                    "separation": None if result.separation is None else round(result.separation, 2),
                    "threshold": round(result.threshold, 4),
                    "mean_wall_clock_ms": round(float(group["wall_clock_ms"].mean()), 1),
                }
            )
    return pd.DataFrame(rows)


def plot_scenario_series(series: pd.DataFrame, cfg, scenario: str, subset: str) -> Path:
    meta = load_scenario(scenario, cfg)[2]
    data = series[(series["scenario"] == scenario) & (series["subset"] == subset)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for detector, group in data.groupby("detector"):
        group = group.sort_values("window_id")
        ax.plot(group["window_id"], group["score"], lw=1.3, label=detector)
    if meta["kind"] != "none":
        ax.axvline(meta["drift_start_window"], color="red", ls="--", lw=1)
    ax.set_xlabel("window")
    ax.set_ylabel("drift score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{scenario} — {subset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOTS_DIR / f"bench_{scenario}_{subset}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join(
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def write_report(results: pd.DataFrame, thresholds: dict, cfg, held_out, evaluation) -> None:
    sections = [
        "# QuantumGuard Phase 4 benchmark report",
        "",
        "Auto-generated by `benchmark/run_benchmark.py`. Interpretation is "
        "maintained in `benchmark/INTERPRETATION.md` and embedded below.",
        "",
        "## Protocol",
        "",
        f"- Held-out scenarios (threshold calibration, feature-map selection): {', '.join(held_out)}",
        f"- Evaluation scenarios (final numbers, disjoint — asserted): {', '.join(evaluation)}",
        f"- Threshold = pre-drift mean + {cfg.benchmark.threshold_sigmas} sigma on held-out pre-drift windows, "
        f"per detector x subset; detection requires {cfg.benchmark.detection_persistence} consecutive "
        "windows above threshold.",
        f"- Deployed quantum feature map (selected on held-out, frozen before evaluation): "
        f"`{cfg.quantum_kernel.feature_map}`.",
        "",
        "## Detection performance — population view",
        "",
        to_markdown_table(
            results[results["subset"] == "population"][
                ["scenario", "kind", "detector", "detected", "delay", "false_alarm_rate", "separation"]
            ]
        ),
        "",
        "## Detection performance — fraud-subset view",
        "",
        to_markdown_table(
            results[results["subset"] == "fraud"][
                ["scenario", "kind", "detector", "detected", "delay", "false_alarm_rate", "separation"]
            ]
        ),
        "",
        "## Compute cost (mean ms per window, population windows)",
        "",
        to_markdown_table(
            results[results["subset"] == "population"]
            .groupby("detector", as_index=False)["mean_wall_clock_ms"]
            .mean()
            .round(1)
            .sort_values("mean_wall_clock_ms")
        ),
        "",
        "## Calibrated thresholds",
        "",
        "```json",
        json.dumps(thresholds, indent=2),
        "```",
        "",
    ]
    if INTERPRETATION_PATH.exists():
        sections += ["## Interpretation", "", INTERPRETATION_PATH.read_text(encoding="utf-8"), ""]
    REPORT_PATH.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    cfg = load_config()
    held_out, evaluation = assert_benchmark_split(cfg)
    all_scenarios = held_out + evaluation
    PLOTS_DIR.mkdir(exist_ok=True)

    if SERIES_PATH.exists() and "--recompute" not in sys.argv:
        series = pd.read_csv(SERIES_PATH)
        missing = set(all_scenarios) - set(series["scenario"].unique())
        if missing:
            raise SystemExit(f"cached series missing scenarios {missing}; rerun with --recompute")
        print(f"loaded cached score series from {SERIES_PATH}")
    else:
        series = compute_score_series(cfg, all_scenarios)
        series.to_csv(SERIES_PATH, index=False)

    thresholds = calibrate(series, cfg, held_out)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    results = evaluate(series, thresholds, cfg, evaluation)
    results.to_csv(RESULTS_PATH, index=False)

    for scenario in evaluation:
        for subset in ("population", "fraud"):
            plot_scenario_series(series, cfg, scenario, subset)

    write_report(results, thresholds, cfg, held_out, evaluation)
    print(f"\nwrote {RESULTS_PATH}, {THRESHOLDS_PATH}, {REPORT_PATH}")
    print(results[results["subset"] == "population"].to_string(index=False))
