from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quantumguard.config import Config, load_config


class PCAReducer:
    """Standardize + PCA, fit ONCE on reference data and then applied to every
    window. Shared by the quantum kernel and the classical baselines so they all
    see identical inputs."""

    def __init__(self, n_components: int | None = None, config: Config | None = None):
        cfg = config if config is not None else load_config()
        self.n_components = n_components if n_components is not None else cfg.pca.n_components
        self._scaler = StandardScaler()
        self._pca: PCA | None = None

    def fit(self, reference: np.ndarray) -> "PCAReducer":
        reference = np.asarray(reference, dtype=float)
        n_components = min(self.n_components, reference.shape[1], len(reference))
        self._pca = PCA(n_components=n_components, random_state=0)
        self._pca.fit(self._scaler.fit_transform(reference))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAReducer must be fit on reference data before transform")
        return self._pca.transform(self._scaler.transform(np.asarray(X, dtype=float)))

    def fit_transform(self, reference: np.ndarray) -> np.ndarray:
        return self.fit(reference).transform(reference)


def subsample_window(X: np.ndarray, max_samples: int, seed: int = 0) -> np.ndarray:
    """Deterministic uniform subsample of a window down to max_samples rows."""
    X = np.asarray(X)
    if len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[np.sort(idx)]


def prepare_windows(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    n_components: int,
    max_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-reduce (fit on reference only) and subsample both windows."""
    reducer = PCAReducer(n_components=n_components)
    ref_reduced = reducer.fit_transform(np.asarray(reference, dtype=float))
    cur_reduced = reducer.transform(np.asarray(current, dtype=float))
    return (
        subsample_window(ref_reduced, max_samples, seed),
        subsample_window(cur_reduced, max_samples, seed + 1),
    )


def prepare_windows_from_config(
    reference: np.ndarray,
    current: np.ndarray,
    config: Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The single window-preparation path shared by the quantum kernel AND the
    classical baselines, so every multivariate detector sees identical inputs."""
    cfg = config if config is not None else load_config()
    return prepare_windows(
        reference,
        current,
        n_components=min(cfg.pca.n_components, cfg.quantum_kernel.max_qubits),
        max_samples=cfg.quantum_kernel.max_window_samples,
        seed=cfg.quantum_kernel.subsample_seed,
    )


def iter_windows(
    X: np.ndarray,
    window_size: int | None = None,
    stride: int | None = None,
    config: Config | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (window_id, window) slices over a stream. Trailing partial windows
    are dropped so every window has identical sample size."""
    cfg = config if config is not None else load_config()
    window_size = window_size if window_size is not None else cfg.windowing.window_size
    stride = stride if stride is not None else cfg.windowing.stride
    X = np.asarray(X)
    window_id = 0
    for start in range(0, len(X) - window_size + 1, stride):
        yield window_id, X[start : start + window_size]
        window_id += 1
