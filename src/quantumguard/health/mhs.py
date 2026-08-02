from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from quantumguard.config import Config, load_config
from quantumguard.drift.classical import ClassicalDriftDetector
from quantumguard.drift.quantum_kernel import QuantumKernelDriftDetector


@dataclass(frozen=True)
class HealthComponents:
    """One window's health snapshot. cds/qds are the values FUSED into MHS
    (max of population and fraud subset when use_fraud_subset_max is on); the
    raw per-subset scores are kept alongside for the audit trail."""

    window_id: int
    cds: float
    qds: float
    pc: float
    at: float
    mhs: float
    status: str
    cds_population: float
    cds_fraud: float | None
    qds_population: float
    qds_fraud: float | None


def status_for(mhs: float, bands: Config) -> str:
    if mhs >= bands.healthy_min:
        return "healthy"
    if mhs < bands.critical_max:
        return "critical"
    return "warning"


class ModelHealthScorer:
    """Streams windows and produces the fused Model Health Score.

    Deterministic given (model, reference, window sequence) — no LLM calls,
    per CLAUDE.md. Ground-truth labels are simulated to arrive
    `label_latency_windows` after a window is scored, so the accuracy trend
    (AT) always runs on lagged labels only.
    """

    def __init__(
        self,
        model,
        *,
        feature_columns: list[str],
        drift_columns: list[str],
        label_column: str = "isFraud",
        config: Config | None = None,
    ):
        cfg = config if config is not None else load_config()
        self._cfg = cfg
        self._model = model
        self._feature_columns = feature_columns
        self._drift_columns = drift_columns
        self._label_column = label_column

        weights = cfg.mhs.weights
        if abs(weights.cds + weights.qds + weights.pc + weights.at - 1.0) > 1e-9:
            raise ValueError("mhs.weights must sum to 1.0")
        self._weights = weights
        self._bands = cfg.mhs.status_bands
        self._label_lag = cfg.mhs.label_latency_windows
        self._at_recent = cfg.mhs.at_recent_windows
        self._use_fraud_max = cfg.mhs.use_fraud_subset_max
        self._min_fraud_n = cfg.classical_drift.min_fraud_subset_n

        self._classical = ClassicalDriftDetector(cfg)
        self._quantum = QuantumKernelDriftDetector(config=cfg)

        self._ref_X: np.ndarray | None = None
        self._ref_fraud_mask: np.ndarray | None = None
        self._ref_confidence: float | None = None
        self._ref_auc: float | None = None
        self._pending_labels: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._labeled_aucs: list[float] = []

    # -- setup -----------------------------------------------------------

    def fit_reference(self, reference: pd.DataFrame) -> "ModelHealthScorer":
        self._ref_X = reference[self._drift_columns].to_numpy(dtype=float)
        self._ref_fraud_mask = (reference[self._label_column] == 1).to_numpy()

        proba = self._model.predict_proba(reference[self._feature_columns])[:, 1]
        self._ref_confidence = self._rescaled_confidence(proba)
        self._ref_auc = float(
            roc_auc_score(reference[self._label_column].to_numpy(), proba)
        )
        return self

    def set_model(self, model, reference: pd.DataFrame) -> "ModelHealthScorer":
        """Swap the monitored model (e.g. after an admin deploys a candidate)
        and re-baseline PC/AT against the reference so the new model is judged
        on its own confidence and accuracy profile."""
        self._model = model
        proba = model.predict_proba(reference[self._feature_columns])[:, 1]
        self._ref_confidence = self._rescaled_confidence(proba)
        self._ref_auc = float(roc_auc_score(reference[self._label_column].to_numpy(), proba))
        self._pending_labels.clear()
        self._labeled_aucs.clear()
        return self

    @staticmethod
    def _rescaled_confidence(proba: np.ndarray) -> float:
        """Mean distance from the decision boundary, rescaled from [0.5, 1] to [0, 1]."""
        return float(2.0 * (np.mean(np.maximum(proba, 1.0 - proba)) - 0.5))

    # -- per-window scoring ---------------------------------------------

    def _drift_pair(self, cur_X: np.ndarray, cur_fraud_mask: np.ndarray) -> tuple[
        float, float | None, float, float | None
    ]:
        cds_pop = self._classical.score(self._ref_X, cur_X)
        qds_pop = self._quantum.score(self._ref_X, cur_X)

        cds_fraud = qds_fraud = None
        if (
            cur_fraud_mask.sum() >= self._min_fraud_n
            and self._ref_fraud_mask.sum() >= self._min_fraud_n
        ):
            fraud_result = self._classical.result(
                self._ref_X,
                cur_X,
                reference_mask=self._ref_fraud_mask,
                current_mask=cur_fraud_mask,
            )
            if not fraud_result.low_confidence:
                cds_fraud = fraud_result.combined
            qds_fraud = self._quantum.score(
                self._ref_X[self._ref_fraud_mask], cur_X[cur_fraud_mask]
            )
        return cds_pop, cds_fraud, qds_pop, qds_fraud

    def _accuracy_trend(self, window_id: int) -> float:
        """Release labels for windows older than the simulated lag, then score
        health as recent labeled-window AUC relative to reference AUC."""
        due = [entry for entry in self._pending_labels if entry[0] <= window_id - self._label_lag]
        self._pending_labels = [
            entry for entry in self._pending_labels if entry[0] > window_id - self._label_lag
        ]
        for _, y_true, y_proba in due:
            if len(np.unique(y_true)) == 2:
                self._labeled_aucs.append(float(roc_auc_score(y_true, y_proba)))

        if not self._labeled_aucs:
            return 1.0  # no labeled evidence yet -> assume healthy
        recent = np.mean(self._labeled_aucs[-self._at_recent :])
        return float(np.clip(recent / max(self._ref_auc, 1e-6), 0.0, 1.0))

    def score_window(self, window: pd.DataFrame, window_id: int) -> HealthComponents:
        if self._ref_X is None:
            raise RuntimeError("call fit_reference before score_window")

        cur_X = window[self._drift_columns].to_numpy(dtype=float)
        cur_fraud_mask = (window[self._label_column] == 1).to_numpy()
        cds_pop, cds_fraud, qds_pop, qds_fraud = self._drift_pair(cur_X, cur_fraud_mask)

        if self._use_fraud_max:
            cds = max(cds_pop, cds_fraud or 0.0)
            qds = max(qds_pop, qds_fraud or 0.0)
        else:
            cds, qds = cds_pop, qds_pop

        proba = self._model.predict_proba(window[self._feature_columns])[:, 1]
        pc = float(
            np.clip(self._rescaled_confidence(proba) / max(self._ref_confidence, 1e-6), 0.0, 1.0)
        )

        self._pending_labels.append(
            (window_id, window[self._label_column].to_numpy(), proba)
        )
        at = self._accuracy_trend(window_id)

        w = self._weights
        mhs = float(
            np.clip(w.cds * (1.0 - cds) + w.qds * (1.0 - qds) + w.pc * pc + w.at * at, 0.0, 1.0)
        )

        return HealthComponents(
            window_id=window_id,
            cds=cds,
            qds=qds,
            pc=pc,
            at=at,
            mhs=mhs,
            status=status_for(mhs, self._bands),
            cds_population=cds_pop,
            cds_fraud=cds_fraud,
            qds_population=qds_pop,
            qds_fraud=qds_fraud,
        )
