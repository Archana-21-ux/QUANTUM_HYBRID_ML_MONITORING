from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

from quantumguard.config import Config, load_config


@dataclass(frozen=True)
class ClassicalDriftResult:
    """Per-window classical drift scores, each mean-aggregated over features
    and normalized to [0, 1]."""

    psi: float
    kl: float
    ks: float
    hellinger: float
    combined: float
    n_reference: int
    n_current: int
    low_confidence: bool


def _quantile_bin_edges(reference_col: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges from reference quantiles. Outermost bins are open-ended,
    so current values outside the reference range still land in a bin."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(reference_col, quantiles))


def _binned_probs(col: np.ndarray, inner_edges: np.ndarray, smoothing: float) -> np.ndarray:
    bin_idx = np.searchsorted(inner_edges, col, side="right")
    counts = np.bincount(bin_idx, minlength=len(inner_edges) + 1).astype(float)
    counts += smoothing
    return counts / counts.sum()


def classical_drift_scores(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int,
    smoothing: float,
    psi_cap: float,
    kl_cap: float,
    min_subset_n: int = 2,
    reference_mask: np.ndarray | None = None,
    current_mask: np.ndarray | None = None,
) -> ClassicalDriftResult:
    """Compute PSI, KL, KS, and Hellinger between reference and current windows.

    Each statistic is computed per feature against reference-quantile bins
    (Laplace-smoothed so sparse/empty bins never produce inf), mean-aggregated
    over features, and normalized to [0, 1]. Optional boolean masks restrict
    either side to a subset (e.g. fraud-labeled rows); when a masked side falls
    below `min_subset_n` the result is zeroed and flagged low-confidence rather
    than reporting an unstable estimate.
    """
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    current = np.atleast_2d(np.asarray(current, dtype=float))
    if reference_mask is not None:
        reference = reference[np.asarray(reference_mask, dtype=bool)]
    if current_mask is not None:
        current = current[np.asarray(current_mask, dtype=bool)]

    n_ref, n_cur = len(reference), len(current)
    if n_ref < max(2, min_subset_n) or n_cur < max(2, min_subset_n):
        return ClassicalDriftResult(
            psi=0.0, kl=0.0, ks=0.0, hellinger=0.0, combined=0.0,
            n_reference=n_ref, n_current=n_cur, low_confidence=True,
        )
    if reference.shape[1] != current.shape[1]:
        raise ValueError(
            f"feature mismatch: reference has {reference.shape[1]}, current has {current.shape[1]}"
        )

    psi_vals, kl_vals, ks_vals, hellinger_vals = [], [], [], []
    for j in range(reference.shape[1]):
        ref_col, cur_col = reference[:, j], current[:, j]
        inner_edges = _quantile_bin_edges(ref_col, n_bins)
        p = _binned_probs(ref_col, inner_edges, smoothing)
        q = _binned_probs(cur_col, inner_edges, smoothing)

        log_ratio = np.log(p / q)
        psi_vals.append(float(np.sum((p - q) * log_ratio)))
        kl_vals.append(float(np.sum(q * np.log(q / p))))
        hellinger_vals.append(float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q))))))
        ks_vals.append(float(ks_2samp(ref_col, cur_col).statistic))

    psi_score = min(float(np.mean(psi_vals)) / psi_cap, 1.0)
    kl_score = min(float(np.mean(kl_vals)) / kl_cap, 1.0)
    ks_score = float(np.mean(ks_vals))
    hellinger_score = float(np.mean(hellinger_vals))
    combined = float(np.mean([psi_score, kl_score, ks_score, hellinger_score]))

    return ClassicalDriftResult(
        psi=max(psi_score, 0.0),
        kl=max(kl_score, 0.0),
        ks=ks_score,
        hellinger=hellinger_score,
        combined=max(combined, 0.0),
        n_reference=n_ref,
        n_current=n_cur,
        low_confidence=False,
    )


class ClassicalDriftDetector:
    """CDS detector: mean of the four normalized classical statistics."""

    name = "classical_cds"

    def __init__(self, config: Config | None = None):
        cfg = config if config is not None else load_config()
        cd = cfg.classical_drift
        self._n_bins = cd.n_bins
        self._smoothing = cd.laplace_smoothing
        self._psi_cap = cd.normalization.psi_cap
        self._kl_cap = cd.normalization.kl_cap
        self._min_subset_n = cd.min_fraud_subset_n

    def result(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        *,
        reference_mask: np.ndarray | None = None,
        current_mask: np.ndarray | None = None,
    ) -> ClassicalDriftResult:
        subset = reference_mask is not None or current_mask is not None
        return classical_drift_scores(
            reference,
            current,
            n_bins=self._n_bins,
            smoothing=self._smoothing,
            psi_cap=self._psi_cap,
            kl_cap=self._kl_cap,
            min_subset_n=self._min_subset_n if subset else 2,
            reference_mask=reference_mask,
            current_mask=current_mask,
        )

    def score(self, reference: np.ndarray, current: np.ndarray) -> float:
        return self.result(reference, current).combined
