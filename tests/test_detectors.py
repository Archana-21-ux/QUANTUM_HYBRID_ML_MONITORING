"""Every multivariate detector (quantum feature maps + classical baselines)
passes the same synthetic-drift battery: protocol compliance, near-zero on
no-shift, clear rise on shift, wall-clock recorded, caps enforced."""

import copy

import numpy as np
import pytest

from quantumguard.config import Config, load_config
from quantumguard.drift.base import DriftDetector
from quantumguard.drift.baselines import DomainClassifierDetector, MMDRBFDetector
from quantumguard.drift.quantum_kernel import QuantumKernelDriftDetector

RNG = np.random.default_rng(123)
N_FEATURES = 8

REFERENCE = RNG.normal(0.0, 1.0, size=(800, N_FEATURES))
CURRENT_SAME = RNG.normal(0.0, 1.0, size=(400, N_FEATURES))
CURRENT_SHIFTED = RNG.normal(0.75, 1.0, size=(400, N_FEATURES))

DETECTOR_FACTORIES = {
    "quantum_angle": lambda: QuantumKernelDriftDetector("angle"),
    "quantum_zz": lambda: QuantumKernelDriftDetector("zz"),
    "quantum_reupload": lambda: QuantumKernelDriftDetector("reupload"),
    "mmd_rbf": lambda: MMDRBFDetector(),
    "domain_classifier": lambda: DomainClassifierDetector(),
}


@pytest.fixture(scope="module", params=list(DETECTOR_FACTORIES))
def detector(request):
    return DETECTOR_FACTORIES[request.param]()


def test_implements_protocol(detector):
    assert isinstance(detector, DriftDetector)
    assert isinstance(detector.name, str) and detector.name


def test_no_shift_score_near_zero(detector):
    score = detector.score(REFERENCE, CURRENT_SAME)
    assert 0.0 <= score <= 1.0
    assert score < 0.2, f"{detector.name} scored {score:.3f} on identical distributions"


def test_score_rises_on_shift(detector):
    base = detector.score(REFERENCE, CURRENT_SAME)
    drifted = detector.score(REFERENCE, CURRENT_SHIFTED)
    assert drifted > base + 0.15, (
        f"{detector.name}: shifted={drifted:.3f} not clearly above base={base:.3f}"
    )


def test_wall_clock_recorded(detector):
    detector.score(REFERENCE, CURRENT_SAME)
    assert detector.last_wall_clock_ms is not None
    assert detector.last_wall_clock_ms > 0.0


def test_qubit_cap_enforced():
    cfg = Config(copy.deepcopy(dict(load_config())))
    cfg["quantum_kernel"]["max_qubits"] = 4
    detector = QuantumKernelDriftDetector("angle", config=cfg)
    # 8 PCA components requested, but the cap must win: window prep reduces to 4
    score = detector.score(REFERENCE, CURRENT_SHIFTED)
    assert 0.0 <= score <= 1.0
    assert set(detector._qnode_cache) == {4}

    with pytest.raises(ValueError, match="exceeds configured cap"):
        detector._qnode(5)


def test_unknown_feature_map_rejected():
    with pytest.raises(ValueError, match="unknown feature map"):
        QuantumKernelDriftDetector("bell_state_magic")
