from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from quantumguard.config import Config, load_config
from quantumguard.models.registry import (
    get_deployed_entry,
    get_model_path,
    next_version,
    register_model,
)
from quantumguard.models.train_baseline import build_pipeline, evaluate
from quantumguard.storage import db


@dataclass(frozen=True)
class RetrainResult:
    candidate_version: str
    deployed_version: str
    candidate_id: int  # retrain_candidates row, status 'pending'
    offline_metrics: dict[str, Any]
    n_recent: int
    n_historical: int


def blend_training_data(
    recent: pd.DataFrame,
    historical_pool: pd.DataFrame,
    *,
    recent_fraction: float,
    historical_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, int, int]:
    """Blend so recent:historical row counts follow the configured fractions,
    keeping ALL recent rows and sampling the historical pool to match."""
    n_recent = len(recent)
    n_hist = min(
        int(round(n_recent * historical_fraction / recent_fraction)), len(historical_pool)
    )
    hist_sample = historical_pool.sample(n=n_hist, random_state=random_state)
    blend = pd.concat([recent, hist_sample], ignore_index=True)
    return blend.sample(frac=1.0, random_state=random_state).reset_index(drop=True), n_recent, n_hist


def run_retraining(
    stream: pd.DataFrame,
    historical_pool: pd.DataFrame,
    *,
    trigger_window: int,
    retrain_window_width: int,
    feature_columns: list[str],
    label_column: str,
    trend_decision_id: int | None,
    conn: sqlite3.Connection,
    config: Config | None = None,
    compare_versions: list[str] | None = None,
) -> RetrainResult:
    """Triggered retraining: blend recent + historical data, train a candidate,
    evaluate candidate AND deployed on a holdout carved from the RECENT
    (drifted-regime) data, version + register the candidate, and record it as a
    PENDING approval in the DB. Never deploys anything itself.
    """
    cfg = config if config is not None else load_config()
    rt = cfg.retrain
    random_state = cfg.baseline_model.random_state

    lo = trigger_window - retrain_window_width + 1
    recent = stream[(stream["window_id"] >= lo) & (stream["window_id"] <= trigger_window)]
    recent = recent[feature_columns + [label_column]]
    if len(recent) < rt.min_recent_rows:
        raise ValueError(
            f"only {len(recent)} recent rows in windows [{lo}, {trigger_window}]; "
            f"need at least {rt.min_recent_rows}"
        )

    recent_train, recent_eval = train_test_split(
        recent,
        test_size=rt.eval_holdout_fraction,
        random_state=random_state,
        stratify=recent[label_column],
    )

    blend, n_recent, n_hist = blend_training_data(
        recent_train,
        historical_pool[feature_columns + [label_column]],
        recent_fraction=rt.recent_data_fraction,
        historical_fraction=rt.historical_data_fraction,
        random_state=random_state,
    )

    candidate = build_pipeline(cfg)
    candidate.fit(blend[feature_columns], blend[label_column])

    deployed_entry = get_deployed_entry(cfg)
    deployed = joblib.load(get_model_path(deployed_entry["version"], cfg))

    X_eval, y_eval = recent_eval[feature_columns], recent_eval[label_column]
    candidate_metrics = evaluate(candidate, X_eval, y_eval)
    deployed_metrics = evaluate(deployed, X_eval, y_eval)

    # sliding-window comparison: score retained past versions on the SAME
    # recent-regime holdout so all contenders are judged on identical data
    comparison = {}
    for version in compare_versions or []:
        past_model = joblib.load(get_model_path(version, cfg))
        comparison[version] = evaluate(past_model, X_eval, y_eval)

    offline_metrics = {
        "eval_set": "recent_regime_holdout",
        "candidate": candidate_metrics,
        "deployed": deployed_metrics,
        "deployed_version": deployed_entry["version"],
        "deltas": {
            key: round(candidate_metrics[key] - deployed_metrics[key], 4)
            for key in ("fraud_precision", "fraud_recall", "fraud_f1", "roc_auc", "average_precision")
        },
        "blend": {
            "n_recent": n_recent,
            "n_historical": n_hist,
            "recent_windows": [int(lo), int(trigger_window)],
        },
        "comparison": comparison,
    }

    version = next_version(cfg)
    model_filename = f"Model_{version}.pkl"
    model_dir = cfg.path("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, model_dir / model_filename)
    register_model(
        version=version,
        model_filename=model_filename,
        metrics=candidate_metrics,
        trained_on=f"retrain@window{trigger_window} recent[{lo},{trigger_window}]+historical",
        config=cfg,
    )

    candidate_id = db.insert_retrain_candidate(
        conn,
        model_version=version,
        trend_decision_id=trend_decision_id,
        offline_metrics=offline_metrics,
    )

    return RetrainResult(
        candidate_version=version,
        deployed_version=deployed_entry["version"],
        candidate_id=candidate_id,
        offline_metrics=offline_metrics,
        n_recent=n_recent,
        n_historical=n_hist,
    )
