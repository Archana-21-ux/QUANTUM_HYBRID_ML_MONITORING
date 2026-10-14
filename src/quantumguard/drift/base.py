from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DriftDetector(Protocol):
    """Every drift detector in QuantumGuard implements this protocol.

    `score` compares a reference window against a current window and returns
    a drift score in [0, 1]: ~0 means no drift, values near 1 mean severe drift.
    Both arrays are 2-D (n_samples, n_features) with matching feature columns.
    """

    name: str

    def score(self, reference: np.ndarray, current: np.ndarray) -> float: ...
