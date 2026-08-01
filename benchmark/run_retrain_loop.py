"""Phase 5 done-criterion: the full loop, headless.

drift injected -> MHS degrades -> trend trigger -> candidate retrained ->
offline candidate-vs-deployed metrics -> shadow replay -> PENDING approval
record (never auto-deployed).

Usage: uv run python benchmark/run_retrain_loop.py [scenario_name]
"""

from __future__ import annotations

import sys

import joblib

from quantumguard.config import load_config
from quantumguard.drift.injection import NUMERIC_FEATURES, load_scenario
from quantumguard.health.mhs import ModelHealthScorer
from quantumguard.health.trend import evaluate_trend
from quantumguard.models.registry import get_deployed_entry, get_model_path
from quantumguard.models.train_baseline import CATEGORICAL_FEATURES
from quantumguard.narration.narrate import narrate
from quantumguard.retrain.pipeline import run_retraining
from quantumguard.retrain.shadow import run_shadow
from quantumguard.storage import db

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def main(scenario_name: str = "sudden_covariate_shift_v1") -> None:
    cfg = load_config()
    reference, stream, meta = load_scenario(scenario_name, cfg)
    deployed_entry = get_deployed_entry(cfg)
    model = joblib.load(get_model_path(deployed_entry["version"], cfg))
    print(f"scenario={scenario_name}  deployed={deployed_entry['version']}")

    scorer = ModelHealthScorer(
        model, feature_columns=FEATURES, drift_columns=NUMERIC_FEATURES, config=cfg
    ).fit_reference(reference)

    db.init_db()
    mhs_series: list[float] = []

    with db.get_connection() as conn:
        for window_id, window in stream.groupby("window_id"):
            components = scorer.score_window(window, int(window_id))
            mhs_series.append(components.mhs)
            decision = evaluate_trend(mhs_series, cfg, window_id=int(window_id))
            if not decision.triggered:
                continue

            print(f"\n[window {window_id}] TRIGGER  {narrate(decision, components)}")
            decision_id = db.insert_trend_decision(
                conn,
                window_id=int(window_id),
                slope=decision.slope,
                p_value=decision.p_value,
                cusum_stat=decision.cusum_stat,
                triggered=True,
                retrain_window_width=decision.retrain_window_width,
                series=list(decision.series),
            )

            result = run_retraining(
                stream,
                historical_pool=reference,
                trigger_window=int(window_id),
                retrain_window_width=decision.retrain_window_width,
                feature_columns=FEATURES,
                label_column="isFraud",
                trend_decision_id=decision_id,
                conn=conn,
                config=cfg,
            )
            print(
                f"candidate {result.candidate_version} trained on "
                f"{result.n_recent} recent + {result.n_historical} historical rows"
            )
            deltas = result.offline_metrics["deltas"]
            print(
                "offline (recent-regime holdout) candidate-vs-deployed deltas: "
                + ", ".join(f"{k}={v:+.3f}" for k, v in deltas.items())
            )

            shadow_stream = stream[stream["window_id"] > window_id]
            summary = run_shadow(
                shadow_stream,
                candidate_id=result.candidate_id,
                candidate_version=result.candidate_version,
                deployed_version=result.deployed_version,
                feature_columns=FEATURES,
                label_column="isFraud",
                conn=conn,
                config=cfg,
            )
            print(
                f"shadow: {summary.n_windows} windows, agreement "
                f"{summary.overall_agreement:.3f}, fraud caught only by candidate: "
                f"{summary.fraud_caught_only_by_candidate}, only by deployed: "
                f"{summary.fraud_caught_only_by_deployed}"
            )

            pending = db.get_pending_candidates(conn)
            print(
                f"pending approvals in DB: "
                f"{[(row['id'], row['model_version'], row['status']) for row in pending]}"
            )
            print("loop complete — candidate awaits admin approval (Phase 6 dashboard).")
            return

    print("stream ended without a trigger")


if __name__ == "__main__":
    main(*sys.argv[1:])
