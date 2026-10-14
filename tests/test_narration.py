import pytest

from quantumguard.health.mhs import HealthComponents
from quantumguard.health.trend import evaluate_trend
from quantumguard.narration.narrate import narrate, narrate_llm


def make_components(**overrides) -> HealthComponents:
    fields = dict(
        window_id=60,
        cds=0.62,
        qds=0.55,
        pc=0.9,
        at=0.85,
        mhs=0.58,
        status="warning",
        cds_population=0.05,
        cds_fraud=0.62,
        qds_population=0.5,
        qds_fraud=0.55,
    )
    fields.update(overrides)
    return HealthComponents(**fields)


def test_triggered_narration_states_evidence_and_width():
    series = [0.9] * 20 + [0.55, 0.45, 0.42]
    decision = evaluate_trend(series)
    assert decision.triggered

    text = narrate(decision)
    assert "Retraining was triggered" in text
    assert "sudden" in text
    assert str(decision.retrain_window_width) in text
    assert f"{decision.slope:.4f}" in text or "CUSUM" in text


def test_healthy_narration():
    decision = evaluate_trend([0.9] * 15)
    text = narrate(decision)
    assert "no retraining" in text.lower()
    assert "0.90" in text


def test_fraud_concentration_called_out():
    decision = evaluate_trend([0.9] * 20 + [0.6, 0.55, 0.52])
    text = narrate(decision, components=make_components())
    assert "fraud-labeled subset" in text
    assert "0.62" in text


def test_narration_is_deterministic():
    decision = evaluate_trend([0.9] * 20 + [0.5, 0.45, 0.44])
    assert narrate(decision, make_components()) == narrate(decision, make_components())


def test_llm_narration_disabled_by_default():
    decision = evaluate_trend([0.9] * 15)
    with pytest.raises(RuntimeError, match="llm_enabled"):
        narrate_llm(decision)


def test_llm_prompt_grounds_on_frozen_decision():
    """The LLM prompt must carry every decision fact plus the canonical
    template — narration is downstream-only, so the record IS the input."""
    from quantumguard.narration.narrate import build_llm_prompt

    decision = evaluate_trend([0.9] * 20 + [0.55, 0.45, 0.42])
    prompt = build_llm_prompt(decision, make_components())
    assert f"window_id: {decision.window_id}" in prompt
    assert "triggered: True" in prompt
    assert f"retrain_window_width: {decision.retrain_window_width}" in prompt
    assert "Retraining was triggered" in prompt  # the template rendering rides along
    assert "0.62" in prompt  # fraud-subset cds from components
