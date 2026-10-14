from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quantumguard.config import Config, load_config
from quantumguard.models.registry import get_model_path
from quantumguard.storage import db


@dataclass(frozen=True)
class ShadowSummary:
    candidate_version: str
    deployed_version: str
    n_windows: int
    overall_agreement: float
    fraud_caught_only_by_candidate: int
    fraud_caught_only_by_deployed: int
    per_window: list[dict[str, Any]] = field(repr=False, default_factory=list)


def run_shadow(
    stream: pd.DataFrame,
    *,
    candidate_id: int,
    candidate_version: str,
    deployed_version: str,
    feature_columns: list[str],
    label_column: str,
    conn: sqlite3.Connection,
    config: Config | None = None,
    candidate_model=None,
    deployed_model=None,
) -> ShadowSummary:
    """Replay windows through candidate AND deployed on identical traffic.

    Logs one shadow_logs row per window: agreement rate plus the divergence
    cases where a true fraud was caught by exactly one of the two models —
    the rows an admin actually needs to inspect before approving. The
    candidate's predictions never leave this harness (shadow, not serving).
    """
    cfg = config if config is not None else load_config()
    threshold = cfg.shadow.prediction_threshold
    max_logged = cfg.shadow.max_divergences_logged_per_window
    n_windows = cfg.shadow.replay_windows

    if candidate_model is None:
        candidate_model = joblib.load(get_model_path(candidate_version, cfg))
    if deployed_model is None:
        deployed_model = joblib.load(get_model_path(deployed_version, cfg))

    window_ids = sorted(stream["window_id"].unique())[:n_windows]
    per_window = []
    candidate_only = deployed_only = 0

    for window_id in window_ids:
        window = stream[stream["window_id"] == window_id]
        X = window[feature_columns]
        y = window[label_column].to_numpy()

        proba_candidate = candidate_model.predict_proba(X)[:, 1]
        proba_deployed = deployed_model.predict_proba(X)[:, 1]
        pred_candidate = proba_candidate >= threshold
        pred_deployed = proba_deployed >= threshold

        agreement = float(np.mean(pred_candidate == pred_deployed))

        disagree = pred_candidate != pred_deployed
        fraud_divergence = disagree & (y == 1)
        candidate_only += int((fraud_divergence & pred_candidate).sum())
        deployed_only += int((fraud_divergence & pred_deployed).sum())

        cases = []
        for idx in np.flatnonzero(fraud_divergence)[:max_logged]:
            cases.append(
                {
                    "row": int(window.index[idx]),
                    "caught_by": "candidate" if pred_candidate[idx] else "deployed",
                    "proba_candidate": round(float(proba_candidate[idx]), 4),
                    "proba_deployed": round(float(proba_deployed[idx]), 4),
                }
            )

        db.insert_shadow_log(
            conn,
            candidate_id=candidate_id,
            window_id=int(window_id),
            agreement_rate=agreement,
            divergence_cases=cases,
        )
        per_window.append(
            {"window_id": int(window_id), "agreement": agreement, "n_divergent_fraud": len(cases)}
        )

    overall = float(np.mean([w["agreement"] for w in per_window])) if per_window else 1.0
    return ShadowSummary(
        candidate_version=candidate_version,
        deployed_version=deployed_version,
        n_windows=len(per_window),
        overall_agreement=overall,
        fraud_caught_only_by_candidate=candidate_only,
        fraud_caught_only_by_deployed=deployed_only,
        per_window=per_window,
    )
