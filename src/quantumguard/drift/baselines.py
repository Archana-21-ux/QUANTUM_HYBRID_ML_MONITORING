from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from quantumguard.config import Config, load_config
from quantumguard.drift.preprocess import prepare_windows_from_config


class MMDRBFDetector:
    """Maximum Mean Discrepancy with an RBF kernel (median-heuristic bandwidth),
    on the same PCA-reduced subsampled windows as the quantum kernel. Biased
    MMD^2 estimate mapped to [0, 1] via the configured score cap."""

    name = "mmd_rbf"

    def __init__(self, config: Config | None = None):
        cfg = config if config is not None else load_config()
        self._cfg = cfg
        self._score_cap = cfg.baselines.mmd_rbf.score_cap
        self.last_wall_clock_ms: float | None = None

    def score(self, reference: np.ndarray, current: np.ndarray) -> float:
        t0 = time.perf_counter()
        ref, cur = prepare_windows_from_config(reference, current, self._cfg)

        combined = np.vstack([ref, cur])
        sq_dists = np.sum(
            (combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1
        )
        median_sq = np.median(sq_dists[np.triu_indices_from(sq_dists, k=1)])
        gamma = 1.0 / max(median_sq, 1e-12)

        kernel = np.exp(-gamma * sq_dists)
        n_ref = len(ref)
        k_rr = kernel[:n_ref, :n_ref]
        k_cc = kernel[n_ref:, n_ref:]
        k_rc = kernel[:n_ref, n_ref:]

        def offdiag_mean(K: np.ndarray) -> float:
            n = len(K)
            return float((K.sum() - np.trace(K)) / (n * (n - 1)))

        mmd2 = offdiag_mean(k_rr) + offdiag_mean(k_cc) - 2.0 * float(k_rc.mean())
        score = float(np.clip(mmd2 / self._score_cap, 0.0, 1.0))
        self.last_wall_clock_ms = (time.perf_counter() - t0) * 1000.0
        return score


class DomainClassifierDetector:
    """Train a classifier to distinguish reference rows from current rows;
    cross-validated AUC maps to a drift score: AUC 0.5 (indistinguishable) -> 0,
    AUC 1.0 (fully separable) -> 1."""

    name = "domain_classifier"

    def __init__(self, config: Config | None = None):
        cfg = config if config is not None else load_config()
        self._cfg = cfg
        self._n_folds = cfg.baselines.domain_classifier.n_folds
        model = cfg.baselines.domain_classifier.model
        if model != "logistic_regression":
            raise ValueError(f"unsupported domain-classifier model {model!r}")
        self.last_wall_clock_ms: float | None = None

    def score(self, reference: np.ndarray, current: np.ndarray) -> float:
        t0 = time.perf_counter()
        ref, cur = prepare_windows_from_config(reference, current, self._cfg)

        X = np.vstack([ref, cur])
        y = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])
        cv = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=0)
        proba = cross_val_predict(
            LogisticRegression(max_iter=1000), X, y, cv=cv, method="predict_proba"
        )[:, 1]
        auc = roc_auc_score(y, proba)

        score = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))
        self.last_wall_clock_ms = (time.perf_counter() - t0) * 1000.0
        return score
