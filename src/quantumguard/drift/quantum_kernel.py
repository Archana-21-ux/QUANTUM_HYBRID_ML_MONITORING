from __future__ import annotations

import time

import numpy as np
import pennylane as qml

from quantumguard.config import Config, load_config
from quantumguard.drift.preprocess import prepare_windows_from_config

FEATURE_MAPS = ("angle", "zz", "reupload")


class QuantumKernelDriftDetector:
    """Quantum-inspired kernel drift detector.

    Windows are PCA-reduced and subsampled (shared prep with the classical
    baselines), scaled to rotation angles, and embedded as statevectors via a
    PennyLane feature map. The fidelity kernel K(x, y) = |<phi(x)|phi(y)>|^2 is
    computed by simulating one state per SAMPLE and taking inner products in a
    single matmul — never one circuit per pair, which would be O(n^2) circuit
    evaluations for the same result on a noiseless simulator.

    Drift score = 1 - (mean cross-kernel similarity) / (mean within-reference
    similarity), clipped to [0, 1]: identical distributions give ~0 because the
    cross similarity matches the reference self-similarity.

    The qubit / layer / sample caps are methodological (kernel concentration:
    fidelity kernels flatten toward 0 as qubit count grows), not just
    performance caps — enforced here from config, never raised silently.
    """

    def __init__(self, map_name: str | None = None, config: Config | None = None):
        cfg = config if config is not None else load_config()
        qk = cfg.quantum_kernel
        self.map_name = map_name if map_name is not None else qk.feature_map
        if self.map_name not in FEATURE_MAPS:
            raise ValueError(f"unknown feature map {self.map_name!r}, expected one of {FEATURE_MAPS}")
        self.name = f"quantum_{self.map_name}"
        self._cfg = cfg
        self.max_qubits = qk.max_qubits
        self.n_layers = min(qk.max_layers, 2)
        self.max_window_samples = qk.max_window_samples
        self.angle_clip_sigmas = qk.angle_clip_sigmas
        self.last_wall_clock_ms: float | None = None
        self._qnode_cache: dict[int, qml.QNode] = {}

    # -- circuit ---------------------------------------------------------

    def _qnode(self, n_qubits: int) -> qml.QNode:
        if n_qubits > self.max_qubits:
            raise ValueError(f"{n_qubits} qubits exceeds configured cap of {self.max_qubits}")
        if n_qubits not in self._qnode_cache:
            dev = qml.device("default.qubit", wires=n_qubits)
            wires = list(range(n_qubits))
            map_name, n_layers = self.map_name, self.n_layers

            @qml.qnode(dev)
            def circuit(x):
                if map_name == "angle":
                    qml.AngleEmbedding(x, wires=wires, rotation="Y")
                elif map_name == "zz":
                    qml.IQPEmbedding(x, wires=wires, n_repeats=n_layers)
                else:  # reupload
                    for _ in range(n_layers):
                        qml.AngleEmbedding(x, wires=wires, rotation="Y")
                        if n_qubits > 1:
                            for i in wires:
                                qml.CNOT(wires=[i, (i + 1) % n_qubits])
                        qml.AngleEmbedding(x, wires=wires, rotation="Z")
                return qml.state()

            self._qnode_cache[n_qubits] = circuit
        return self._qnode_cache[n_qubits]

    def _states(self, angles: np.ndarray) -> np.ndarray:
        circuit = self._qnode(angles.shape[1])
        try:  # parameter broadcasting: one batched simulation for the whole window
            states = np.asarray(circuit(angles))
            if states.shape[0] != len(angles):
                raise ValueError("broadcast shape mismatch")
        except Exception:
            states = np.stack([np.asarray(circuit(x)) for x in angles])
        return states

    # -- scoring ---------------------------------------------------------

    def _to_angles(self, ref: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map PCA components (centered on reference) into [-pi, pi], clipping
        at angle_clip_sigmas reference stds so outliers cannot wrap the circle."""
        sigma = ref.std(axis=0)
        sigma[sigma == 0] = 1.0
        scale = self.angle_clip_sigmas * sigma

        def to_angle(Z: np.ndarray) -> np.ndarray:
            return np.pi * np.clip(Z / scale, -1.0, 1.0)

        return to_angle(ref), to_angle(cur)

    def score(self, reference: np.ndarray, current: np.ndarray) -> float:
        t0 = time.perf_counter()
        ref_p, cur_p = prepare_windows_from_config(reference, current, self._cfg)
        ref_a, cur_a = self._to_angles(ref_p, cur_p)

        states_ref = self._states(ref_a)
        states_cur = self._states(cur_a)

        k_rr = np.abs(states_ref @ states_ref.conj().T) ** 2
        k_rc = np.abs(states_ref @ states_cur.conj().T) ** 2

        n = len(states_ref)
        within_ref = (k_rr.sum() - np.trace(k_rr).real) / (n * (n - 1))
        cross = float(k_rc.mean())

        score = 0.0 if within_ref <= 0 else float(np.clip(1.0 - cross / within_ref, 0.0, 1.0))
        self.last_wall_clock_ms = (time.perf_counter() - t0) * 1000.0
        return score
