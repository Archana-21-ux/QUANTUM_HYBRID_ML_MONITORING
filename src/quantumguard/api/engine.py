from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import joblib

from dataclasses import asdict

from quantumguard.api.ws import ConnectionManager
from quantumguard.config import Config, load_config
from quantumguard.drift.feature_breakdown import FeatureDriftAnalyzer
from quantumguard.drift.injection import NUMERIC_FEATURES, load_scenario
from quantumguard.health.mhs import ModelHealthScorer
from quantumguard.health.relative_trigger import RelativeRetrainTrigger
from quantumguard.models.registry import get_deployed_entry, get_model_path, get_recent_deployments
from quantumguard.models.train_baseline import CATEGORICAL_FEATURES
from quantumguard.retrain.comparison import comparison_log_line, pick_winner
from quantumguard.retrain.pipeline import run_retraining
from quantumguard.retrain.shadow import run_shadow
from quantumguard.storage import db

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SimulationEngine:
    """Streams a drift scenario window-by-window for the live dashboard.

    Each window: score MHS (population + fraud subset), persist via db.py,
    evaluate the trend, and broadcast typed events. On a trigger it retrains a
    candidate, shadow-validates it, and leaves it PENDING — deployment only
    happens through the approve endpoint, which flips the registry flag and
    asks this engine to hot-reload the deployed model mid-stream.
    """

    def __init__(self, ws_manager: ConnectionManager, config: Config | None = None):
        self._cfg = config if config is not None else load_config()
        self._ws = ws_manager
        self._task: asyncio.Task | None = None
        self._reload_requested = False
        self.status: dict[str, Any] = {"state": "idle", "scenario": None, "window_id": None}

    # -- control ---------------------------------------------------------

    def request_model_reload(self) -> None:
        self._reload_requested = True

    async def start(self, scenario: str) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError("simulation already running")
        self._ws.reset_replay()
        self._task = asyncio.create_task(self._run(scenario))

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.status = {"state": "idle", "scenario": None, "window_id": None}
        await self._emit({"type": "simulation", "status": dict(self.status)})

    # -- internals -------------------------------------------------------

    async def _emit(self, event: dict[str, Any]) -> None:
        event["ts"] = _now()
        await self._ws.broadcast(event)

    async def _reasoning(self, text: str, kind: str = "info") -> None:
        await self._emit({"type": "reasoning", "kind": kind, "text": text})

    def _load_deployed(self):
        entry = get_deployed_entry(self._cfg)
        return entry["version"], joblib.load(get_model_path(entry["version"], self._cfg))

    async def _run(self, scenario: str) -> None:
        cfg = self._cfg
        sim = cfg.simulation
        try:
            reference, stream, meta = await asyncio.to_thread(load_scenario, scenario, cfg)
            deployed_version, model = await asyncio.to_thread(self._load_deployed)
            scorer = ModelHealthScorer(
                model, feature_columns=FEATURES, drift_columns=NUMERIC_FEATURES, config=cfg
            )
            await asyncio.to_thread(scorer.fit_reference, reference)
            # per-feature drift analyzer shares the same reference sample
            analyzer = FeatureDriftAnalyzer(cfg)
            await asyncio.to_thread(analyzer.fit_reference, reference)
            await asyncio.to_thread(db.init_db, cfg.path("db_path"))

            self.status = {"state": "running", "scenario": scenario, "window_id": None}
            await self._emit(
                {"type": "simulation", "status": dict(self.status), "meta": dict(meta),
                 "deployed": deployed_version}
            )
            await self._reasoning(
                f"Simulation started on '{scenario}' with deployed model {deployed_version}."
            )

            mhs_series: list[float] = []
            last_status = "healthy"
            # retraining is judged RELATIVE to the deployed model's baseline,
            # captured at the first window after each deployment
            trigger = RelativeRetrainTrigger(cfg)
            baseline_pending = True
            n_windows = int(stream["window_id"].nunique())
            epoch = 0

            while True:
              # each epoch replays the same stream with window ids shifted so
              # charts, retraining and the DB see one ever-increasing timeline
              epoch_stream = stream.copy()
              epoch_stream["window_id"] = epoch_stream["window_id"] + epoch * n_windows

              for window_id, window in epoch_stream.groupby("window_id"):
                window_id = int(window_id)

                if self._reload_requested:
                    self._reload_requested = False
                    deployed_version, model = await asyncio.to_thread(self._load_deployed)
                    await asyncio.to_thread(scorer.set_model, model, reference)
                    await self._emit({"type": "deployed", "version": deployed_version})
                    baseline_pending = True  # requirement 3: re-baseline on deployment

                components, proba_sample, feature_rows = await asyncio.to_thread(
                    self._score_and_persist, scorer, analyzer, window, window_id
                )
                mhs_series.append(components.mhs)

                await self._emit(
                    {
                        "type": "health",
                        "window_id": window_id,
                        "mhs": round(components.mhs, 4),
                        "status": components.status,
                        "cds": round(components.cds, 4),
                        "qds": round(components.qds, 4),
                        "pc": round(components.pc, 4),
                        "at": round(components.at, 4),
                        "cds_population": round(components.cds_population, 4),
                        "cds_fraud": None if components.cds_fraud is None else round(components.cds_fraud, 4),
                        "qds_population": round(components.qds_population, 4),
                        "qds_fraud": None if components.qds_fraud is None else round(components.qds_fraud, 4),
                    }
                )
                await self._emit(
                    {"type": "predictions", "window_id": window_id, "rows": proba_sample}
                )
                await self._emit(
                    {"type": "feature_drift", "window_id": window_id, "rows": feature_rows}
                )
                self.status["window_id"] = window_id

                if components.status != last_status:
                    await self._reasoning(
                        f"Window {window_id}: status changed {last_status} -> {components.status} "
                        f"(MHS {components.mhs:.2f}).",
                        kind="status",
                    )
                    last_status = components.status

                if baseline_pending:
                    # freeze this window's scores as the deployed model's baseline
                    baseline = trigger.set_baseline(deployed_version, window_id, components)
                    await self._reasoning(baseline.describe(), kind="deploy")
                    baseline_pending = False

                decision = trigger.evaluate(window_id, components)
                if decision.triggered:
                    await self._reasoning(decision.reason, kind="trigger")
                    await self._handle_trigger(
                        epoch_stream, reference, decision, window_id, deployed_version
                    )
                elif decision.breached and (
                    decision.sustained_windows == 1
                    or decision.sustained_windows == trigger.sustain_windows
                ):
                    # narrate the start of a breach episode and the moment it
                    # becomes actionable-but-held, without flooding the log
                    await self._reasoning(decision.reason, kind="status")

                await asyncio.sleep(sim.window_delay_seconds)

              if not sim.loop_stream:
                  break
              epoch += 1
              await self._reasoning(
                  f"Stream ended — replaying scenario (pass {epoch + 1}), timeline continues."
              )

            self.status = {"state": "finished", "scenario": scenario, "window_id": self.status["window_id"]}
            await self._emit({"type": "simulation", "status": dict(self.status)})
            await self._reasoning("Simulation finished.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surface engine crashes to the dashboard
            self.status = {"state": "error", "scenario": scenario, "error": str(exc)}
            await self._emit({"type": "simulation", "status": dict(self.status)})
            raise

    def _score_and_persist(self, scorer, analyzer, window, window_id: int):
        components = scorer.score_window(window, window_id)
        # per-feature drift for this window (already sorted by hybrid desc)
        feature_rows = [asdict(row) for row in analyzer.analyze(window)]
        with db.get_connection(self._cfg.path("db_path")) as conn:
            db.insert_health_score(
                conn,
                window_id=window_id,
                cds=components.cds,
                qds=components.qds,
                pc=components.pc,
                at=components.at,
                mhs=components.mhs,
                status=components.status,
            )
            db.insert_feature_drift(conn, window_id=window_id, rows=feature_rows)

        sample = window.sample(
            n=min(self._cfg.simulation.predictions_sample_size, len(window)), random_state=window_id
        )
        proba = scorer._model.predict_proba(sample[FEATURES])[:, 1]
        rows = [
            {
                "type": row.type,
                "amount": round(float(row.amount), 2),
                "proba": round(float(p), 4),
                "is_fraud": int(row.isFraud),
            }
            for row, p in zip(sample.itertuples(), proba)
        ]
        return components, rows, feature_rows

    async def _handle_trigger(self, stream, reference, decision, window_id, deployed_version):
        mc = self._cfg.model_comparison
        delta_keys = ("fraud_precision", "fraud_recall", "fraud_f1", "roc_auc", "average_precision")

        def retrain_and_compare():
            with db.get_connection(self._cfg.path("db_path")) as conn:
                # at most one pending candidate — newer proposals win
                superseded = [dict(row) for row in db.get_pending_candidates(conn)]
                for row in superseded:
                    db.decide_retrain_candidate(
                        conn,
                        candidate_id=row["id"],
                        status="rejected",
                        decided_by="system (superseded)",
                    )

                # sliding buffer: last N previously deployed versions
                past_versions = get_recent_deployments(
                    mc.buffer_size, self._cfg, exclude=deployed_version
                )
                result = run_retraining(
                    stream,
                    historical_pool=reference,
                    trigger_window=window_id,
                    retrain_window_width=decision.retrain_window_width,
                    feature_columns=FEATURES,
                    label_column="isFraud",
                    trend_decision_id=None,  # relative-trigger audit lives in the reasoning log
                    conn=conn,
                    config=self._cfg,
                    compare_versions=past_versions,
                )

                # every contender was scored on the same recent-regime holdout
                scores = {
                    result.candidate_version: result.offline_metrics["candidate"],
                    result.deployed_version: result.offline_metrics["deployed"],
                    **result.offline_metrics["comparison"],
                }
                winner = pick_winner(scores, metric=mc.metric, tiebreak=mc.tiebreak_metric)

                rollback = False
                proposal_id = result.candidate_id
                proposal_version = result.candidate_version
                proposal_metrics = result.offline_metrics
                if winner == result.deployed_version:
                    # deployed model still best: no proposal at all
                    db.decide_retrain_candidate(
                        conn,
                        candidate_id=result.candidate_id,
                        status="rejected",
                        decided_by="system (deployed model outperformed)",
                    )
                    proposal_id = None
                elif winner != result.candidate_version:
                    # a retained past version wins: propose a ROLLBACK instead,
                    # still approval-gated like any other candidate
                    rollback = True
                    db.decide_retrain_candidate(
                        conn,
                        candidate_id=result.candidate_id,
                        status="rejected",
                        decided_by=f"system (rollback to {winner} preferred)",
                    )
                    proposal_metrics = {
                        **result.offline_metrics,
                        "rollback": True,
                        "replaces_candidate": result.candidate_version,
                        "candidate": scores[winner],  # 'candidate' column = rollback target
                        "deltas": {
                            key: round(scores[winner][key] - scores[result.deployed_version][key], 4)
                            for key in delta_keys
                        },
                    }
                    proposal_version = winner
                    proposal_id = db.insert_retrain_candidate(
                        conn,
                        model_version=winner,
                        trend_decision_id=None,
                        offline_metrics=proposal_metrics,
                    )

                summary = None
                if proposal_id is not None:
                    summary = run_shadow(
                        stream[stream["window_id"] > window_id],
                        candidate_id=proposal_id,
                        candidate_version=proposal_version,
                        deployed_version=result.deployed_version,
                        feature_columns=FEATURES,
                        label_column="isFraud",
                        conn=conn,
                        config=self._cfg,
                    )
            return (result, past_versions, scores, winner, rollback,
                    proposal_id, proposal_version, proposal_metrics, summary, superseded)

        await self._reasoning("Retraining candidate model on the configured recent/historical blend…")
        (result, past_versions, scores, winner, rollback, proposal_id,
         proposal_version, proposal_metrics, summary, superseded) = await asyncio.to_thread(
            retrain_and_compare
        )

        for row in superseded:
            await self._reasoning(
                f"Pending candidate {row['model_version']} superseded by {result.candidate_version}.",
                kind="candidate",
            )
        await self._reasoning(
            comparison_log_line(
                window_id=window_id,
                candidate_version=result.candidate_version,
                deployed_version=result.deployed_version,
                past_versions=past_versions,
                scores=scores,
                winner=winner,
                metric=mc.metric,
            ),
            kind="candidate",
        )

        if proposal_id is None:
            return  # deployed model won — nothing pending

        await self._emit(
            {
                "type": "candidate",
                "candidate_id": proposal_id,
                "version": proposal_version,
                "deployed_version": result.deployed_version,
                "offline_metrics": proposal_metrics,
                "shadow": {
                    "n_windows": summary.n_windows,
                    "overall_agreement": round(summary.overall_agreement, 4),
                    "fraud_caught_only_by_candidate": summary.fraud_caught_only_by_candidate,
                    "fraud_caught_only_by_deployed": summary.fraud_caught_only_by_deployed,
                },
            }
        )
        proposal_kind = "Rollback to" if rollback else "Candidate"
        await self._reasoning(
            f"{proposal_kind} {proposal_version} ready: shadow agreement "
            f"{summary.overall_agreement:.1%}, frauds caught only by proposal "
            f"{summary.fraud_caught_only_by_candidate} vs only by deployed "
            f"{summary.fraud_caught_only_by_deployed}. Awaiting admin approval.",
            kind="candidate",
        )
