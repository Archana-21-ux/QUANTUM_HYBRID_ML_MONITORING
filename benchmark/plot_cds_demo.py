"""Phase 1 done-criterion demo: stream a drift scenario window-by-window,
compute the classical drift score (CDS) on population and fraud subset, and
plot the trajectories against the known drift start.

Usage: uv run python benchmark/plot_cds_demo.py [scenario_name ...]
Defaults to one drifted scenario and the no-drift control. PNGs land in
benchmark/plots/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantumguard.config import load_config
from quantumguard.drift.classical import ClassicalDriftDetector
from quantumguard.drift.injection import NUMERIC_FEATURES, load_scenario

PLOTS_DIR = Path(__file__).parent / "plots"


def compute_cds_series(scenario_name: str, cfg) -> dict:
    reference, stream, meta = load_scenario(scenario_name, cfg)
    detector = ClassicalDriftDetector(cfg)

    ref_X = reference[NUMERIC_FEATURES].to_numpy(dtype=float)
    ref_fraud_mask = (reference["isFraud"] == 1).to_numpy()

    window_ids, population, fraud_subset = [], [], []
    for window_id, window in stream.groupby("window_id"):
        cur_X = window[NUMERIC_FEATURES].to_numpy(dtype=float)
        cur_fraud_mask = (window["isFraud"] == 1).to_numpy()

        pop = detector.result(ref_X, cur_X)
        sub = detector.result(
            ref_X, cur_X, reference_mask=ref_fraud_mask, current_mask=cur_fraud_mask
        )
        window_ids.append(window_id)
        population.append(pop.combined)
        # low-confidence subset windows (too few fraud rows) plot as gaps, not zeros
        fraud_subset.append(np.nan if sub.low_confidence else sub.combined)

    return {
        "window_ids": np.array(window_ids),
        "population": np.array(population),
        "fraud_subset": np.array(fraud_subset),
        "meta": meta,
    }


def plot_series(series: dict, out_path: Path) -> None:
    meta = series["meta"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series["window_ids"], series["population"], label="CDS (population)", lw=1.8)
    ax.plot(
        series["window_ids"], series["fraud_subset"], label="CDS (fraud subset)", lw=1.2, alpha=0.8
    )
    if meta["kind"] != "none":
        ax.axvline(
            meta["drift_start_window"], color="red", ls="--", lw=1, label="drift injected"
        )
        if meta["kind"] == "gradual":
            ax.axvline(
                meta["drift_start_window"] + meta["gradual_ramp_windows"],
                color="orange", ls=":", lw=1, label="ramp complete",
            )
    ax.set_xlabel("window")
    ax.set_ylabel("classical drift score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{meta['name']}  (kind={meta['kind']}, scale={meta['scale']})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    names = sys.argv[1:] or ["sudden_covariate_shift_v1", "gradual_covariate_shift_v1",
                             "fraud_subset_shift_v1", "no_drift_control_v1"]
    PLOTS_DIR.mkdir(exist_ok=True)
    for name in names:
        series = compute_cds_series(name, cfg)
        out = PLOTS_DIR / f"{name}_cds.png"
        plot_series(series, out)
        pre = series["population"][series["window_ids"] < series["meta"]["drift_start_window"]]
        post = series["population"][series["window_ids"] >= series["meta"]["drift_start_window"]]
        print(
            f"{name}: pre-drift CDS mean={pre.mean():.3f}, post-drift CDS mean={post.mean():.3f} "
            f"-> {out}"
        )
