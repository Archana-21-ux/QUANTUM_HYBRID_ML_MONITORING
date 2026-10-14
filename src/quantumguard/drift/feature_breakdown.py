"""Feature-level drift breakdown.

The aggregate CDS/QDS answer "is the data drifting?"; this module answers
"WHICH input feature is drifting?". For every feature declared in
configs/default.yaml (feature_breakdown.features) it computes:

- classical_drift: the same four statistics the aggregate CDS fuses
  (PSI, KL, KS, Hellinger), applied to that ONE column only, each normalized
  to [0, 1] and mean-fused — so a single feature's score is directly
  comparable to the aggregate score.
- quantum_drift: the same angle-map fidelity kernel used for the aggregate
  QDS, run on that one column (one qubit; the categorical feature is one-hot
  encoded first, one qubit per category).
- hybrid_drift: the plain mean of the two, used for ranking and the
  severity status dot.

Deterministic given (reference, window) — no randomness beyond the fixed-seed
subsampling shared with the aggregate detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from quantumguard.config import Config, load_config
from quantumguard.drift.classical import _binned_probs, _quantile_bin_edges
from quantumguard.drift.quantum_kernel import QuantumKernelDriftDetector


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    classical_drift: float
    quantum_drift: float
    hybrid_drift: float
    severity: str  # "ok" | "warn" | "crit" — from config bands on hybrid


class FeatureDriftAnalyzer:
    """Per-feature drift scores between a fixed reference sample and each
    incoming window. fit_reference() once, then analyze() per window."""

    def __init__(self, config: Config | None = None):
        cfg = config if config is not None else load_config()
        self._cfg = cfg
        self._features = [dict(f) for f in cfg.feature_breakdown.features]
        self._warn = cfg.feature_breakdown.severity_warn
        self._crit = cfg.feature_breakdown.severity_crit
        cd = cfg.classical_drift
        self._n_bins = cd.n_bins
        self._smoothing = cd.laplace_smoothing
        self._psi_cap = cd.normalization.psi_cap
        self._kl_cap = cd.normalization.kl_cap
        # one shared angle-map detector: 1 qubit for a numeric column,
        # n_categories qubits for the one-hot categorical column
        self._quantum = QuantumKernelDriftDetector("angle", cfg)
        self._reference: pd.DataFrame | None = None
        self._categories: dict[str, list] = {}

    # -- feature extraction ----------------------------------------------

    def _extract(self, frame: pd.DataFrame, spec: dict) -> np.ndarray:
        """Pull one feature out of a window as a 2-D numeric matrix the
        detectors can consume."""
        column, kind = spec["column"], spec["kind"]
        if kind == "hour_of_day":
            # PaySim's `step` counts simulation HOURS from the start, so the
            # hour-of-day cycle is simply step mod 24.
            return (frame[column].to_numpy(dtype=float) % 24).reshape(-1, 1)
        if kind == "categorical":
            # One-hot on the category vocabulary FIXED from the reference
            # sample, so reference and window matrices always share columns.
            categories = self._categories[column]
            values = frame[column].to_numpy()
            return np.column_stack([(values == c).astype(float) for c in categories])
        return frame[column].to_numpy(dtype=float).reshape(-1, 1)

    # -- classical score for one feature ---------------------------------

    def _classical_numeric(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Fused classical score for one numeric column: identical recipe to
        the aggregate CDS, minus the across-features mean (there is only one
        feature here)."""
        # bins come from reference quantiles; Laplace smoothing keeps every
        # bin non-zero so the log-based statistics can never return inf
        edges = _quantile_bin_edges(ref, self._n_bins)
        p = _binned_probs(ref, edges, self._smoothing)
        q = _binned_probs(cur, edges, self._smoothing)
        return self._fuse(p, q, ks=float(ks_2samp(ref, cur).statistic))

    def _classical_categorical(self, ref: np.ndarray, cur: np.ndarray, categories: list) -> float:
        """Same fusion for the categorical feature: the 'bins' are simply the
        category frequencies. KS assumes an ordering, which categories don't
        have, so its slot is filled by total variation distance
        (0.5 * sum|p - q|) — also bounded in [0, 1]."""
        def probs(values: np.ndarray) -> np.ndarray:
            counts = np.array([(values == c).sum() for c in categories], dtype=float)
            counts += self._smoothing
            return counts / counts.sum()

        p, q = probs(ref), probs(cur)
        return self._fuse(p, q, ks=float(0.5 * np.abs(p - q).sum()))

    def _fuse(self, p: np.ndarray, q: np.ndarray, *, ks: float) -> float:
        """PSI and KL are unbounded, so they are capped to [0, 1] with the
        same config caps the aggregate CDS uses; KS/TVD and Hellinger are
        naturally in [0, 1]. The fused score is the mean of the four."""
        log_ratio = np.log(p / q)
        psi = min(max(float(np.sum((p - q) * log_ratio)), 0.0) / self._psi_cap, 1.0)
        kl = min(max(float(np.sum(q * np.log(q / p))), 0.0) / self._kl_cap, 1.0)
        hellinger = float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))
        return float(np.mean([psi, kl, ks, hellinger]))

    # -- public API -------------------------------------------------------

    def fit_reference(self, reference: pd.DataFrame) -> "FeatureDriftAnalyzer":
        self._reference = reference
        for spec in self._features:
            if spec["kind"] == "categorical":
                # vocabulary frozen from reference so one-hot shapes stay stable
                self._categories[spec["column"]] = sorted(
                    reference[spec["column"]].unique().tolist()
                )
        return self

    def _severity(self, hybrid: float) -> str:
        if hybrid >= self._crit:
            return "crit"
        if hybrid >= self._warn:
            return "warn"
        return "ok"

    def analyze(self, window: pd.DataFrame) -> list[FeatureDrift]:
        """Score every tracked feature on one window; sorted by hybrid drift
        descending so the most-drifting feature is always row one."""
        if self._reference is None:
            raise RuntimeError("call fit_reference before analyze")

        rows = []
        for spec in self._features:
            ref_matrix = self._extract(self._reference, spec)
            cur_matrix = self._extract(window, spec)

            if spec["kind"] == "categorical":
                classical = self._classical_categorical(
                    self._reference[spec["column"]].to_numpy(),
                    window[spec["column"]].to_numpy(),
                    self._categories[spec["column"]],
                )
            else:
                classical = self._classical_numeric(ref_matrix[:, 0], cur_matrix[:, 0])

            # the quantum detector consumes the same single-feature matrix the
            # classical stats saw; hybrid is the plain mean of the two views
            quantum = self._quantum.score(ref_matrix, cur_matrix)
            hybrid = (classical + quantum) / 2.0

            rows.append(
                FeatureDrift(
                    feature=spec["name"],
                    classical_drift=round(classical, 4),
                    quantum_drift=round(quantum, 4),
                    hybrid_drift=round(hybrid, 4),
                    severity=self._severity(hybrid),
                )
            )

        return sorted(rows, key=lambda row: row.hybrid_drift, reverse=True)
