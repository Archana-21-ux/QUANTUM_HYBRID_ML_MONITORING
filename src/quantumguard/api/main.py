from __future__ import annotations

import io
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from quantumguard.api.engine import FEATURES, SimulationEngine
from quantumguard.api.ws import ConnectionManager
from quantumguard.config import REPO_ROOT, Config, load_config
from quantumguard.models.registry import get_deployed_entry, get_model_path, mark_deployed
from quantumguard.storage import db

DASHBOARD_DIST = REPO_ROOT / "dashboard" / "dist"


class Transaction(BaseModel):
    type: str
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


class PredictRequest(BaseModel):
    transactions: list[Transaction]


class SimulationRequest(BaseModel):
    scenario: str | None = None


class DecisionRequest(BaseModel):
    decided_by: str = "admin"


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config if config is not None else load_config()
    app = FastAPI(title="QuantumGuard", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.api.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = ConnectionManager()
    engine = SimulationEngine(ws_manager, cfg)
    app.state.engine = engine
    db.init_db(cfg.path("db_path"))

    # -- prediction ---------------------------------------------------

    @app.post("/api/predict")
    def predict(request: PredictRequest):
        entry = get_deployed_entry(cfg)
        model = joblib.load(get_model_path(entry["version"], cfg))
        frame = pd.DataFrame([t.model_dump() for t in request.transactions])[FEATURES]
        proba = model.predict_proba(frame)[:, 1]
        return {
            "model_version": entry["version"],
            "predictions": [
                {"fraud_probability": round(float(p), 4), "is_fraud": bool(p >= 0.5)}
                for p in proba
            ],
        }

    # -- monitoring reads ---------------------------------------------

    @app.get("/api/health-score")
    def health_scores(limit: int = 200):
        with db.get_connection(cfg.path("db_path")) as conn:
            rows = db.get_recent_health_scores(conn, limit=limit)
        return {"health_scores": [dict(row) for row in reversed(rows)]}

    @app.get("/api/drift")
    def drift_scores(limit: int = 500):
        with db.get_connection(cfg.path("db_path")) as conn:
            rows = conn.execute(
                "SELECT * FROM drift_scores ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return {"drift_scores": [dict(row) for row in reversed(rows)]}

    @app.get("/api/drift/features")
    def feature_drift():
        """Latest per-feature drift breakdown (most recent scored window)."""
        with db.get_connection(cfg.path("db_path")) as conn:
            rows = db.get_latest_feature_drift(conn)
        return {"feature_drift": [dict(row) for row in rows]}

    @app.get("/api/reasoning-log")
    def reasoning_log(limit: int = 50):
        with db.get_connection(cfg.path("db_path")) as conn:
            rows = conn.execute(
                "SELECT * FROM trend_decisions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        decisions = []
        for row in reversed(rows):
            entry = dict(row)
            entry["series"] = json.loads(entry.pop("series_json"))
            decisions.append(entry)
        return {"trend_decisions": decisions}

    # -- retraining approvals -----------------------------------------

    @app.get("/api/retrain/candidates")
    def candidates():
        with db.get_connection(cfg.path("db_path")) as conn:
            rows = conn.execute(
                "SELECT * FROM retrain_candidates ORDER BY id DESC"
            ).fetchall()
            out = []
            for row in rows:
                entry = dict(row)
                metrics_json = entry.pop("offline_metrics_json")
                entry["offline_metrics"] = json.loads(metrics_json) if metrics_json else None
                logs = db.get_shadow_logs(conn, entry["id"])
                if logs:
                    entry["shadow"] = {
                        "n_windows": len(logs),
                        "overall_agreement": round(
                            sum(l["agreement_rate"] for l in logs) / len(logs), 4
                        ),
                    }
                out.append(entry)
        return {"candidates": out, "deployed_version": get_deployed_entry(cfg)["version"]}

    def _decide(candidate_id: int, status: str, decided_by: str):
        with db.get_connection(cfg.path("db_path")) as conn:
            row = conn.execute(
                "SELECT * FROM retrain_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"candidate {candidate_id} not found")
            if row["status"] != "pending":
                raise HTTPException(409, f"candidate {candidate_id} already {row['status']}")
            db.decide_retrain_candidate(
                conn, candidate_id=candidate_id, status=status, decided_by=decided_by
            )
        return dict(row)

    @app.post("/api/retrain/{candidate_id}/approve")
    async def approve(candidate_id: int, request: DecisionRequest):
        row = _decide(candidate_id, "approved", request.decided_by)
        mark_deployed(row["model_version"], cfg)
        engine.request_model_reload()
        await ws_manager.broadcast(
            {"type": "deployed", "version": row["model_version"], "decided_by": request.decided_by}
        )
        return {"status": "approved", "deployed_version": row["model_version"]}

    @app.post("/api/retrain/{candidate_id}/reject")
    async def reject(candidate_id: int, request: DecisionRequest):
        row = _decide(candidate_id, "rejected", request.decided_by)
        return {"status": "rejected", "model_version": row["model_version"]}

    # -- simulation control -------------------------------------------

    @app.post("/api/simulation/start")
    async def simulation_start(request: SimulationRequest):
        scenario = request.scenario or cfg.simulation.default_scenario
        scenario_dir = cfg.path("drift_scenarios_dir") / scenario
        if not scenario_dir.exists():
            raise HTTPException(404, f"scenario {scenario!r} not generated")
        try:
            await engine.start(scenario)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"status": "started", "scenario": scenario}

    @app.post("/api/simulation/stop")
    async def simulation_stop():
        await engine.stop()
        return {"status": "stopped"}

    @app.get("/api/simulation/status")
    def simulation_status():
        scenarios = sorted(
            p.name for p in cfg.path("drift_scenarios_dir").iterdir() if p.is_dir()
        )
        return {"status": engine.status, "scenarios": scenarios}

    @app.post("/api/scenarios/upload")
    async def upload_scenario(file: UploadFile = File(...)):
        """Turn an uploaded CSV (PaySim schema) into a runnable scenario.

        The first `reference_windows` worth of rows become the drift baseline;
        the remainder becomes the stream, windowed by row order.
        """
        from quantumguard.drift.injection import STREAM_COLUMNS

        required = set(STREAM_COLUMNS) - {"window_id"}
        try:
            frame = pd.read_csv(io.BytesIO(await file.read()))
        except Exception as exc:
            raise HTTPException(400, f"could not parse CSV: {exc}")

        missing = required - set(frame.columns)
        if missing:
            raise HTTPException(
                400, f"CSV is missing required columns: {sorted(missing)}"
            )

        window_size = cfg.windowing.window_size
        ref_windows = cfg.drift_injection.reference_windows
        min_rows = (ref_windows + 5) * window_size  # baseline + at least 5 stream windows
        if len(frame) < min_rows:
            raise HTTPException(
                400,
                f"need at least {min_rows} rows ({ref_windows} reference + 5 stream "
                f"windows of {window_size}); got {len(frame)}",
            )

        slug = re.sub(r"[^a-z0-9]+", "_", (file.filename or "dataset").rsplit(".", 1)[0].lower())
        name = f"custom_{slug}".strip("_")

        reference = frame.iloc[: ref_windows * window_size].copy()
        reference.insert(0, "window_id", reference.reset_index(drop=True).index // window_size)
        stream = frame.iloc[ref_windows * window_size :].copy()
        n_windows = len(stream) // window_size
        stream = stream.iloc[: n_windows * window_size]
        stream.insert(0, "window_id", stream.reset_index(drop=True).index // window_size)

        out = cfg.path("drift_scenarios_dir") / name
        out.mkdir(parents=True, exist_ok=True)
        reference[STREAM_COLUMNS].to_csv(out / "reference.csv", index=False)
        stream[STREAM_COLUMNS].to_csv(out / "stream.csv", index=False)
        meta = {
            "name": name,
            "kind": "custom",
            "drift_start_window": None,
            "n_windows": n_windows,
            "window_size": window_size,
            "source_filename": file.filename,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"scenario": name, "n_windows": n_windows}

    # -- websocket ----------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()  # keepalive pings from the client
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)

    # -- dashboard static ---------------------------------------------

    if DASHBOARD_DIST.exists():
        app.mount("/", StaticFiles(directory=DASHBOARD_DIST, html=True), name="dashboard")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
