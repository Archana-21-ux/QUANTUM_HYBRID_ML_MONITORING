"""Phase 3 done-criterion: feed a full drift scenario end-to-end through the
Model Health Scorer and trend detector, verify trigger timing, persist every
window to SQLite via db.py, and plot the MHS trajectory.

Usage: uv run python benchmark/run_health_demo.py [scenario_name ...]
Plots land in benchmark/plots/, rows in the configured SQLite DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantumguard.config import load_config
from quantumguard.drift.injection import NUMERIC_FEATURES, load_scenario
from quantumguard.health.mhs import ModelHealthScorer
from quantumguard.health.trend import evaluate_trend
from quantumguard.models.registry import get_model_path
from quantumguard.models.train_baseline import CATEGORICAL_FEATURES
from quantumguard.narration.narrate import narrate
from quantumguard.storage import db

PLOTS_DIR = Path(__file__).parent / "plots"


def run_scenario(scenario_name: str, cfg, conn) -> dict:
    reference, stream, meta = load_scenario(scenario_name, cfg)
    model = joblib.load(get_model_path())

    scorer = ModelHealthScorer(
        model,
        feature_columns=CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        drift_columns=NUMERIC_FEATURES,
        config=cfg,
    ).fit_reference(reference)

    mhs_series: list[float] = []
    components_series = []
    first_trigger: dict | None = None

    for window_id, window in stream.groupby("window_id"):
        components = scorer.score_window(window, int(window_id))
        components_series.append(components)
        mhs_series.append(components.mhs)

        db.insert_health_score(
            conn,
            window_id=int(window_id),
            cds=components.cds,
            qds=components.qds,
            pc=components.pc,
            at=components.at,
            mhs=components.mhs,
            status=components.status,
        )
        for subset, cds_value, qds_value in (
            ("population", components.cds_population, components.qds_population),
            ("fraud", components.cds_fraud, components.qds_fraud),
        ):
            if cds_value is not None:
                db.insert_drift_score(
                    conn, window_id=int(window_id), detector_name="classical_cds",
                    subset=subset, score=cds_value,
                )
            if qds_value is not None:
                db.insert_drift_score(
                    conn, window_id=int(window_id), detector_name=scorer._quantum.name,
                    subset=subset, score=qds_value,
                )

        decision = evaluate_trend(mhs_series, cfg, window_id=int(window_id))
        if decision.triggered and first_trigger is None:
            db.insert_trend_decision(
                conn,
                window_id=int(window_id),
                slope=decision.slope,
                p_value=decision.p_value,
                cusum_stat=decision.cusum_stat,
                triggered=True,
                retrain_window_width=decision.retrain_window_width,
                series=list(decision.series),
            )
            first_trigger = {"decision": decision, "components": components}

    return {
        "name": scenario_name,
        "meta": meta,
        "mhs_series": mhs_series,
        "components": components_series,
        "first_trigger": first_trigger,
    }


def plot_run(run: dict, out_path: Path, cfg) -> None:
    meta, series = run["meta"], run["mhs_series"]
    bands = cfg.mhs.status_bands
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhspan(bands.healthy_min, 1.0, alpha=0.08, color="green", label="healthy band")
    ax.axhspan(bands.critical_max, bands.healthy_min, alpha=0.08, color="orange")
    ax.axhspan(0.0, bands.critical_max, alpha=0.08, color="red")

    ax.plot(range(len(series)), series, lw=2, label="MHS")
    ax.plot(
        range(len(series)), [c.cds for c in run["components"]], lw=1, alpha=0.6, label="CDS (fused)"
    )
    ax.plot(
        range(len(series)), [c.at for c in run["components"]], lw=1, alpha=0.6, label="AT"
    )

    if meta["kind"] != "none":
        ax.axvline(meta["drift_start_window"], color="red", ls="--", lw=1, label="drift injected")
    if run["first_trigger"] is not None:
        wid = run["first_trigger"]["decision"].window_id
        ax.axvline(wid, color="purple", ls="-", lw=1.5, label=f"retrain trigger (w{wid})")

    ax.set_xlabel("window")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{run['name']}  (kind={meta['kind']})")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    names = sys.argv[1:] or [
        "sudden_covariate_shift_v1",
        "gradual_covariate_shift_v1",
        "fraud_subset_shift_v1",
        "no_drift_control_v1",
    ]
    PLOTS_DIR.mkdir(exist_ok=True)
    db.init_db()

    with db.get_connection() as conn:
        for name in names:
            run = run_scenario(name, cfg, conn)
            plot_run(run, PLOTS_DIR / f"{name}_health.png", cfg)

            drift_start = run["meta"]["drift_start_window"]
            if run["first_trigger"] is None:
                print(f"{name}: no trigger (drift kind={run['meta']['kind']})")
            else:
                decision = run["first_trigger"]["decision"]
                delay = decision.window_id - drift_start
                print(
                    f"{name}: triggered at window {decision.window_id} "
                    f"(drift at {drift_start}, delay {delay}), type={decision.drift_type}, "
                    f"width={decision.retrain_window_width}, reason={decision.trigger_reason}"
                )
                print(f"  narration: {narrate(decision, run['first_trigger']['components'])}")
