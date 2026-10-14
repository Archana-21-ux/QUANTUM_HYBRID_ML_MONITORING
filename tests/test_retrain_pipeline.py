import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from quantumguard.config import Config, load_config
from quantumguard.models.registry import (
    get_deployed_entry,
    get_recent_deployments,
    load_registry,
    mark_deployed,
    next_version,
    register_model,
)
from quantumguard.models.train_baseline import build_pipeline
from quantumguard.retrain.pipeline import blend_training_data, run_retraining
from quantumguard.storage import db

RNG = np.random.default_rng(21)
FEATURES = ["type", "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]


def make_stream(n_windows: int, window_size: int = 200, shift: float = 0.0) -> pd.DataFrame:
    n = n_windows * window_size
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[: int(n * 0.1)] = 1
    is_fraud = RNG.permutation(is_fraud)
    frame = pd.DataFrame(
        {
            "window_id": np.arange(n) // window_size,
            "type": RNG.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], size=n),
            "step": RNG.integers(1, 700, size=n),
            "amount": RNG.exponential(1000, size=n) + shift + is_fraud * (2000 + shift * 5),
            "oldbalanceOrg": RNG.exponential(5000, size=n),
            "newbalanceOrig": RNG.exponential(5000, size=n),
            "oldbalanceDest": RNG.exponential(5000, size=n),
            "newbalanceDest": RNG.exponential(5000, size=n),
            "isFraud": is_fraud,
        }
    )
    return frame


@pytest.fixture()
def tmp_cfg(tmp_path):
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["paths"]["model_dir"] = str(tmp_path / "models")
    cfg["paths"]["registry_path"] = str(tmp_path / "models" / "registry.json")
    cfg["paths"]["db_path"] = str(tmp_path / "test.db")
    cfg["baseline_model"]["n_estimators"] = 20  # keep test training fast
    return cfg


@pytest.fixture()
def deployed_v1(tmp_cfg):
    """Train and register a small v1 'deployed' model on undrifted data."""
    train = make_stream(5)
    model = build_pipeline(tmp_cfg)
    model.fit(train[FEATURES], train["isFraud"])
    model_dir = tmp_cfg.path("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "Model_v1.pkl")
    register_model(
        version="v1", model_filename="Model_v1.pkl", metrics={}, trained_on="test",
        config=tmp_cfg,
    )
    return model


def test_next_version_and_deployed_fallback(tmp_cfg, deployed_v1):
    assert next_version(tmp_cfg) == "v2"
    assert get_deployed_entry(tmp_cfg)["version"] == "v1"  # fallback: first registered
    register_model(version="v2", model_filename="Model_v2.pkl", metrics={},
                   trained_on="test", config=tmp_cfg)
    assert next_version(tmp_cfg) == "v3"
    mark_deployed("v2", tmp_cfg)
    assert get_deployed_entry(tmp_cfg)["version"] == "v2"


def test_deployment_history_buffer(tmp_cfg, deployed_v1):
    for version in ("v2", "v3", "v4"):
        register_model(version=version, model_filename=f"Model_{version}.pkl", metrics={},
                       trained_on="test", config=tmp_cfg)
    # deploy sequence with a repeat: v2, v3, v2, v4 (v4 now current)
    for version in ("v2", "v3", "v2", "v4"):
        mark_deployed(version, tmp_cfg)

    # buffer excludes the current version, dedupes, most recent first
    assert get_recent_deployments(5, tmp_cfg, exclude="v4") == ["v2", "v3"]
    # oldest falls off when the limit is tighter
    assert get_recent_deployments(1, tmp_cfg, exclude="v4") == ["v2"]
    # unknown versions in history are skipped (registry is the source of truth)
    registry = load_registry(tmp_cfg)
    assert registry["deployment_history"] == ["v2", "v3", "v2", "v4"]


def test_run_retraining_compares_past_versions(tmp_cfg, deployed_v1):
    stream = make_stream(30, shift=3000.0)
    db.init_db(tmp_cfg.path("db_path"))
    with db.get_connection(tmp_cfg.path("db_path")) as conn:
        result = run_retraining(
            stream,
            historical_pool=make_stream(10),
            trigger_window=25,
            retrain_window_width=5,
            feature_columns=FEATURES,
            label_column="isFraud",
            trend_decision_id=None,
            conn=conn,
            config=tmp_cfg,
            compare_versions=["v1"],  # retained past version scored on same holdout
        )
    comparison = result.offline_metrics["comparison"]
    assert set(comparison) == {"v1"}
    for key in ("fraud_f1", "roc_auc", "fraud_precision", "fraud_recall"):
        assert 0.0 <= comparison["v1"][key] <= 1.0


def test_blend_fractions():
    recent = make_stream(4)          # 800 rows
    historical = make_stream(10)     # 2000-row pool
    blend, n_recent, n_hist = blend_training_data(
        recent, historical, recent_fraction=0.8, historical_fraction=0.2, random_state=0
    )
    assert n_recent == 800
    assert n_hist == 200  # 800 * 0.2/0.8
    assert len(blend) == 1000


def test_run_retraining_full(tmp_cfg, deployed_v1):
    stream = make_stream(30, shift=3000.0)  # drifted regime
    db.init_db(tmp_cfg.path("db_path"))

    with db.get_connection(tmp_cfg.path("db_path")) as conn:
        result = run_retraining(
            stream,
            historical_pool=make_stream(10),
            trigger_window=25,
            retrain_window_width=5,
            feature_columns=FEATURES,
            label_column="isFraud",
            trend_decision_id=None,
            conn=conn,
            config=tmp_cfg,
        )

        assert result.candidate_version == "v2"
        assert result.deployed_version == "v1"
        assert (tmp_cfg.path("model_dir") / "Model_v2.pkl").exists()

        registry = load_registry(tmp_cfg)
        assert [m["version"] for m in registry["models"]] == ["v1", "v2"]
        assert not registry["models"][-1]["deployed"], "retraining must never auto-deploy"

        metrics = result.offline_metrics
        assert metrics["eval_set"] == "recent_regime_holdout"
        for side in ("candidate", "deployed"):
            assert "fraud_precision" in metrics[side]
            assert "fraud_recall" in metrics[side]
        assert set(metrics["deltas"]) >= {"fraud_precision", "fraud_recall", "fraud_f1"}
        # metrics are computed on recent-regime data the candidate saw in-distribution:
        # it should not be worse than the stale deployed model there
        assert metrics["deltas"]["fraud_f1"] >= -0.05

        pending = db.get_pending_candidates(conn)
        assert len(pending) == 1
        assert pending[0]["model_version"] == "v2"
        assert pending[0]["status"] == "pending"


def test_run_retraining_rejects_too_little_data(tmp_cfg, deployed_v1):
    stream = make_stream(2, window_size=100)  # 200 rows only
    db.init_db(tmp_cfg.path("db_path"))
    with db.get_connection(tmp_cfg.path("db_path")) as conn:
        with pytest.raises(ValueError, match="recent rows"):
            run_retraining(
                stream,
                historical_pool=make_stream(5),
                trigger_window=1,
                retrain_window_width=2,
                feature_columns=FEATURES,
                label_column="isFraud",
                trend_decision_id=None,
                conn=conn,
                config=tmp_cfg,
            )
