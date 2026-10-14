from types import SimpleNamespace

import pytest

from quantumguard.config import load_config
from quantumguard.health.relative_trigger import RelativeRetrainTrigger

# config defaults: drop/rise threshold 0.25, sustain 3, cooldown 20
HEALTHY = SimpleNamespace(mhs=0.95, cds=0.05, qds=0.02)
DEGRADED = SimpleNamespace(mhs=0.55, cds=0.7, qds=0.9)  # drop 0.40, rises 0.65/0.88
MILD = SimpleNamespace(mhs=0.80, cds=0.1, qds=0.05)     # drop 0.15 — below margin


def make_trigger():
    trigger = RelativeRetrainTrigger(load_config())
    trigger.set_baseline("v1", 0, HEALTHY)
    return trigger


def run(trigger, values, start=1):
    return [trigger.evaluate(start + i, v) for i, v in enumerate(values)]


def test_no_breach_within_margins():
    trigger = make_trigger()
    decisions = run(trigger, [HEALTHY, MILD, HEALTHY])
    assert all(not d.breached and not d.triggered for d in decisions)
    assert decisions[0].baseline_version == "v1"
    assert "within v1 baseline margins" in decisions[0].reason


def test_breach_must_sustain_three_windows():
    trigger = make_trigger()
    # two degraded windows then recovery: streak resets, never triggers
    decisions = run(trigger, [DEGRADED, DEGRADED, HEALTHY, DEGRADED])
    assert [d.sustained_windows for d in decisions] == [1, 2, 0, 1]
    assert not any(d.triggered for d in decisions)
    assert "1/3 sustained windows" in decisions[0].reason


def test_triggers_on_third_sustained_window():
    trigger = make_trigger()
    decisions = run(trigger, [DEGRADED, DEGRADED, DEGRADED])
    assert [d.triggered for d in decisions] == [False, False, True]
    final = decisions[-1]
    assert "Quantum drift rose 0.88 above v1 baseline" in final.reason  # dominant signal
    assert "cooldown expired — generating retraining candidate" in final.reason
    assert final.drift_type in ("sudden", "gradual")
    assert final.retrain_window_width is not None


def test_cooldown_blocks_then_allows_retrigger():
    trigger = make_trigger()
    decisions = run(trigger, [DEGRADED] * 30)
    triggered_at = [d.window_id for d in decisions if d.triggered]
    # first trigger at window 3 (3 sustained), next only after 20-window cooldown
    assert triggered_at == [3, 23]
    blocked = decisions[3]  # window 4: still breached, cooldown holds
    assert blocked.breached and not blocked.triggered
    assert blocked.cooldown_remaining == 19
    assert "cooldown active" in blocked.reason


def test_health_drop_alone_triggers():
    trigger = make_trigger()
    drop_only = SimpleNamespace(mhs=0.65, cds=0.06, qds=0.03)  # drop 0.30, no drift rise
    final = run(trigger, [drop_only] * 3)[-1]
    assert final.triggered
    assert "Health degraded 0.30 below v1 baseline" in final.reason


def test_baseline_reset_rebases_the_margins():
    trigger = make_trigger()
    assert run(trigger, [DEGRADED] * 3)[-1].triggered
    # approving a candidate re-baselines at the degraded regime's scores:
    # the same values are now "normal" and must not breach
    trigger.set_baseline("v2", 10, DEGRADED)
    decisions = run(trigger, [DEGRADED] * 5, start=11)
    assert not any(d.breached for d in decisions)
    assert decisions[0].baseline_version == "v2"
    assert trigger.baseline.describe().startswith("v2 deployed at window 10, baseline health=0.55")


def test_sudden_vs_gradual_classification():
    trigger = make_trigger()
    # cliff: healthy -> deep degradation within the sustain span => sudden
    cliff = [HEALTHY, SimpleNamespace(mhs=0.6, cds=0.5, qds=0.5),
             SimpleNamespace(mhs=0.5, cds=0.6, qds=0.6),
             SimpleNamespace(mhs=0.45, cds=0.6, qds=0.6)]
    final = run(trigger, cliff)[-1]
    assert final.triggered and final.drift_type == "sudden"

    trigger2 = make_trigger()
    # plateau: already-degraded but flat MHS across the span => gradual
    flat = SimpleNamespace(mhs=0.62, cds=0.5, qds=0.5)
    final2 = run(trigger2, [flat] * 3)[-1]
    assert final2.triggered and final2.drift_type == "gradual"
    widths = load_config().trend_detector.retrain_window_width
    assert final2.retrain_window_width == widths["gradual"]


def test_evaluate_before_baseline_raises():
    trigger = RelativeRetrainTrigger(load_config())
    with pytest.raises(RuntimeError, match="set_baseline"):
        trigger.evaluate(0, HEALTHY)
