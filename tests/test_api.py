import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantumguard.api.main import create_app
from quantumguard.config import Config, load_config
from quantumguard.models.registry import get_deployed_entry, register_model
from quantumguard.models.train_baseline import build_pipeline
from quantumguard.storage import db

RNG = np.random.default_rng(31)
FEATURES = ["type", "step", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"]


def make_training_frame(n=2000):
    is_fraud = np.zeros(n, dtype=int)
    is_fraud[: n // 10] = 1
    is_fraud = RNG.permutation(is_fraud)
    return pd.DataFrame(
        {
            "type": RNG.choice(["PAYMENT", "TRANSFER"], size=n),
            "step": RNG.integers(1, 700, size=n),
            "amount": RNG.exponential(1000, size=n) + is_fraud * 3000,
            "oldbalanceOrg": RNG.exponential(5000, size=n),
            "newbalanceOrig": RNG.exponential(5000, size=n),
            "oldbalanceDest": RNG.exponential(5000, size=n),
            "newbalanceDest": RNG.exponential(5000, size=n),
            "isFraud": is_fraud,
        }
    )


@pytest.fixture()
def client(tmp_path):
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["paths"]["model_dir"] = str(tmp_path / "models")
    cfg["paths"]["registry_path"] = str(tmp_path / "models" / "registry.json")
    cfg["paths"]["db_path"] = str(tmp_path / "test.db")
    cfg["paths"]["drift_scenarios_dir"] = str(tmp_path / "scenarios")
    (tmp_path / "scenarios").mkdir()
    cfg["baseline_model"]["n_estimators"] = 20

    train = make_training_frame()
    model = build_pipeline(cfg)
    model.fit(train[FEATURES], train["isFraud"])
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, tmp_path / "models" / "Model_v1.pkl")
    register_model(version="v1", model_filename="Model_v1.pkl", metrics={},
                   trained_on="test", config=cfg)

    app = create_app(cfg)
    return TestClient(app), cfg


def make_transaction(amount=100.0):
    return {
        "type": "TRANSFER", "step": 10, "amount": amount,
        "oldbalanceOrg": 500.0, "newbalanceOrig": 400.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 100.0,
    }


def test_predict(client):
    test_client, _ = client
    response = test_client.post(
        "/api/predict", json={"transactions": [make_transaction(), make_transaction(9e6)]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        assert 0.0 <= prediction["fraud_probability"] <= 1.0


def test_health_and_drift_endpoints_empty(client):
    test_client, _ = client
    assert test_client.get("/api/health-score").json() == {"health_scores": []}
    assert test_client.get("/api/drift").json() == {"drift_scores": []}
    assert test_client.get("/api/reasoning-log").json() == {"trend_decisions": []}
    assert test_client.get("/api/drift/features").json() == {"feature_drift": []}


def test_feature_drift_endpoint_returns_latest_window(client):
    test_client, cfg = client
    row = {"feature": "Amount", "classical_drift": 0.61, "quantum_drift": 0.58,
           "hybrid_drift": 0.6, "severity": "crit"}
    with db.get_connection(cfg.path("db_path")) as conn:
        db.insert_feature_drift(conn, window_id=1, rows=[{**row, "hybrid_drift": 0.1}])
        db.insert_feature_drift(conn, window_id=2, rows=[row])

    body = test_client.get("/api/drift/features").json()
    assert len(body["feature_drift"]) == 1
    latest = body["feature_drift"][0]
    assert latest["window_id"] == 2  # only the newest window is returned
    assert latest["hybrid_drift"] == 0.6
    assert latest["severity"] == "crit"

    # a fresh run whose window ids restart LOW must NOT be masked by the stale
    # higher-numbered rows above — latest is the last inserted batch, keyed off
    # insertion order, not the highest window_id
    with db.get_connection(cfg.path("db_path")) as conn:
        db.insert_feature_drift(
            conn, window_id=0,
            rows=[
                {**row, "feature": "Hour", "hybrid_drift": 0.02, "severity": "ok"},
                {**row, "feature": "Amount", "hybrid_drift": 0.05, "severity": "ok"},
            ],
        )
    fresh = test_client.get("/api/drift/features").json()["feature_drift"]
    assert len(fresh) == 2  # only the fresh window-0 batch, not the stale window-2 row
    assert {r["feature"] for r in fresh} == {"Hour", "Amount"}
    assert fresh[0]["feature"] == "Amount"  # sorted by hybrid_drift desc


def test_candidate_approval_flow(client):
    test_client, cfg = client
    # register a candidate model file + pending DB row, as the pipeline would
    model_dir = cfg.path("model_dir")
    joblib.dump(joblib.load(model_dir / "Model_v1.pkl"), model_dir / "Model_v2.pkl")
    register_model(version="v2", model_filename="Model_v2.pkl", metrics={},
                   trained_on="test-retrain", config=cfg)
    with db.get_connection(cfg.path("db_path")) as conn:
        candidate_id = db.insert_retrain_candidate(
            conn, model_version="v2", trend_decision_id=None,
            offline_metrics={"deltas": {"fraud_f1": 0.05}},
        )

    listed = test_client.get("/api/retrain/candidates").json()
    assert listed["deployed_version"] == "v1"
    assert listed["candidates"][0]["status"] == "pending"
    assert listed["candidates"][0]["offline_metrics"]["deltas"]["fraud_f1"] == 0.05

    response = test_client.post(f"/api/retrain/{candidate_id}/approve", json={"decided_by": "tester"})
    assert response.status_code == 200
    assert response.json()["deployed_version"] == "v2"
    assert get_deployed_entry(cfg)["version"] == "v2"

    with db.get_connection(cfg.path("db_path")) as conn:
        row = conn.execute(
            "SELECT * FROM retrain_candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
    assert row["status"] == "approved"
    assert row["decided_by"] == "tester"

    # deciding twice must conflict
    assert test_client.post(f"/api/retrain/{candidate_id}/approve", json={}).status_code == 409


def test_reject_leaves_deployment_unchanged(client):
    test_client, cfg = client
    model_dir = cfg.path("model_dir")
    joblib.dump(joblib.load(model_dir / "Model_v1.pkl"), model_dir / "Model_v2.pkl")
    register_model(version="v2", model_filename="Model_v2.pkl", metrics={},
                   trained_on="test-retrain", config=cfg)
    with db.get_connection(cfg.path("db_path")) as conn:
        candidate_id = db.insert_retrain_candidate(
            conn, model_version="v2", trend_decision_id=None, offline_metrics=None
        )

    response = test_client.post(f"/api/retrain/{candidate_id}/reject", json={})
    assert response.status_code == 200
    assert get_deployed_entry(cfg)["version"] == "v1"


def test_unknown_candidate_404(client):
    test_client, _ = client
    assert test_client.post("/api/retrain/999/approve", json={}).status_code == 404


def test_simulation_start_unknown_scenario_404(client):
    test_client, _ = client
    response = test_client.post("/api/simulation/start", json={"scenario": "nope"})
    assert response.status_code == 404


def make_paysim_csv(n_rows: int) -> bytes:
    frame = pd.DataFrame(
        {
            "type": RNG.choice(["PAYMENT", "TRANSFER"], size=n_rows),
            "step": RNG.integers(1, 700, size=n_rows),
            "amount": RNG.exponential(1000, size=n_rows),
            "oldbalanceOrg": RNG.exponential(5000, size=n_rows),
            "newbalanceOrig": RNG.exponential(5000, size=n_rows),
            "oldbalanceDest": RNG.exponential(5000, size=n_rows),
            "newbalanceDest": RNG.exponential(5000, size=n_rows),
            "isFraud": RNG.integers(0, 2, size=n_rows),
        }
    )
    return frame.to_csv(index=False).encode()


def test_upload_custom_scenario(client):
    test_client, cfg = client
    window_size = cfg.windowing.window_size
    ref_windows = cfg.drift_injection.reference_windows
    csv = make_paysim_csv((ref_windows + 6) * window_size)

    response = test_client.post(
        "/api/scenarios/upload",
        files={"file": ("My Fraud Data.csv", csv, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"] == "custom_my_fraud_data"
    assert body["n_windows"] == 6

    scenario_dir = cfg.path("drift_scenarios_dir") / body["scenario"]
    assert (scenario_dir / "reference.csv").exists()
    assert (scenario_dir / "stream.csv").exists()
    assert (scenario_dir / "meta.json").exists()
    # it shows up in the dropdown source and is startable
    assert body["scenario"] in test_client.get("/api/simulation/status").json()["scenarios"]


def test_upload_rejects_bad_csv(client):
    test_client, cfg = client
    # missing required columns
    response = test_client.post(
        "/api/scenarios/upload", files={"file": ("bad.csv", b"a,b\n1,2\n", "text/csv")}
    )
    assert response.status_code == 400
    assert "missing required columns" in response.json()["detail"]

    # too few rows
    small = make_paysim_csv(cfg.windowing.window_size)
    response = test_client.post(
        "/api/scenarios/upload", files={"file": ("small.csv", small, "text/csv")}
    )
    assert response.status_code == 400
    assert "need at least" in response.json()["detail"]


def test_simulation_status_lists_scenarios(client):
    test_client, cfg = client
    (cfg.path("drift_scenarios_dir") / "fake_scenario").mkdir(parents=True)
    body = test_client.get("/api/simulation/status").json()
    assert body["status"]["state"] == "idle"
    assert "fake_scenario" in body["scenarios"]
