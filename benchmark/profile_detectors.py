"""Risk item #1 check: per-window wall-clock for every detector on REAL
scenario windows, before Phase 4 builds on them.

Usage: uv run python benchmark/profile_detectors.py [scenario_name]
Writes benchmark/timings.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from quantumguard.config import load_config
from quantumguard.drift.baselines import DomainClassifierDetector, MMDRBFDetector
from quantumguard.drift.classical import ClassicalDriftDetector
from quantumguard.drift.injection import NUMERIC_FEATURES, load_scenario
from quantumguard.drift.quantum_kernel import QuantumKernelDriftDetector

N_PROFILE_WINDOWS = 10
OUT_PATH = Path(__file__).parent / "timings.csv"


def main(scenario_name: str = "sudden_covariate_shift_v1") -> None:
    cfg = load_config()
    reference, stream, _ = load_scenario(scenario_name, cfg)
    ref_X = reference[NUMERIC_FEATURES].to_numpy(dtype=float)

    detectors = [
        ClassicalDriftDetector(cfg),
        QuantumKernelDriftDetector("angle", cfg),
        QuantumKernelDriftDetector("zz", cfg),
        QuantumKernelDriftDetector("reupload", cfg),
        MMDRBFDetector(cfg),
        DomainClassifierDetector(cfg),
    ]

    window_ids = sorted(stream["window_id"].unique())[:N_PROFILE_WINDOWS]
    rows = []
    for detector in detectors:
        timings = []
        for window_id in window_ids:
            window = stream[stream["window_id"] == window_id]
            cur_X = window[NUMERIC_FEATURES].to_numpy(dtype=float)
            import time

            t0 = time.perf_counter()
            detector.score(ref_X, cur_X)
            timings.append((time.perf_counter() - t0) * 1000.0)
        timings = np.array(timings)
        rows.append(
            {
                "detector": detector.name,
                "n_windows": len(timings),
                "mean_ms": round(float(timings.mean()), 1),
                "median_ms": round(float(np.median(timings)), 1),
                "max_ms": round(float(timings.max()), 1),
            }
        )
        print(
            f"{detector.name:18s} mean={rows[-1]['mean_ms']:8.1f}ms  "
            f"median={rows[-1]['median_ms']:8.1f}ms  max={rows[-1]['max_ms']:8.1f}ms"
        )

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main(*sys.argv[1:])
