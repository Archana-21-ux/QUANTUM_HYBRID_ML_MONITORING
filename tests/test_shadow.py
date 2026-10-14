import copy
import json

import numpy as np
import pandas as pd
import pytest

from quantumguard.config import Config, load_config
from quantumguard.retrain.shadow import run_shadow
from quantumguard.storage import db

FEATURES = ["amount"]


class ThresholdModel:
    """Stub: flags fraud when amount exceeds its cutoff."""

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = (X["amount"].to_numpy() > self.cutoff).astype(float) * 0.9 + 0.05
        return np.column_stack([1.0 - p1, p1])


@pytest.fixture()
def tmp_cfg(tmp_path):
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["paths"]["db_path"] = str(tmp_path / "test.db")
    cfg["shadow"]["replay_windows"] = 3
    return cfg


def make_stream() -> pd.DataFrame:
    # 3 windows x 4 rows; amounts straddle the two model cutoffs (100 vs 500)
    return pd.DataFrame(
        {
            "window_id": [0] * 4 + [1] * 4 + [2] * 4,
            "amount": [50, 300, 700, 300] * 3,
            "isFraud": [0, 1, 1, 0] * 3,
        }
    )


def test_shadow_agreement_and_divergences(tmp_cfg):
    stream = make_stream()
    db.init_db(tmp_cfg.path("db_path"))

    with db.get_connection(tmp_cfg.path("db_path")) as conn:
        candidate_id = db.insert_retrain_candidate(
            conn, model_version="v2", trend_decision_id=None, offline_metrics=None
        )
        summary = run_shadow(
            stream,
            candidate_id=candidate_id,
            candidate_version="v2",
            deployed_version="v1",
            feature_columns=FEATURES,
            label_column="isFraud",
            conn=conn,
            config=tmp_cfg,
            candidate_model=ThresholdModel(cutoff=100.0),  # catches amounts 300 and 700
            deployed_model=ThresholdModel(cutoff=500.0),   # catches only 700
        )

        # per window: rows 50(agree neg), 300 fraud (candidate only), 700 (agree pos),
        # 300 legit (disagree, not fraud) -> agreement 2/4
        assert summary.n_windows == 3
        assert summary.overall_agreement == pytest.approx(0.5)
        assert summary.fraud_caught_only_by_candidate == 3  # one per window
        assert summary.fraud_caught_only_by_deployed == 0

        logs = db.get_shadow_logs(conn, candidate_id)
        assert len(logs) == 3
        cases = json.loads(logs[0]["divergence_cases_json"])
        # only the FRAUD disagreement is logged, not the legit one
        assert len(cases) == 1
        assert cases[0]["caught_by"] == "candidate"
        assert cases[0]["proba_candidate"] > cases[0]["proba_deployed"]


def test_shadow_respects_replay_window_cap(tmp_cfg):
    stream = make_stream()
    tmp_cfg["shadow"]["replay_windows"] = 2
    db.init_db(tmp_cfg.path("db_path"))
    with db.get_connection(tmp_cfg.path("db_path")) as conn:
        candidate_id = db.insert_retrain_candidate(
            conn, model_version="v2", trend_decision_id=None, offline_metrics=None
        )
        summary = run_shadow(
            stream,
            candidate_id=candidate_id,
            candidate_version="v2",
            deployed_version="v1",
            feature_columns=FEATURES,
            label_column="isFraud",
            conn=conn,
            config=tmp_cfg,
            candidate_model=ThresholdModel(100.0),
            deployed_model=ThresholdModel(500.0),
        )
        assert summary.n_windows == 2
