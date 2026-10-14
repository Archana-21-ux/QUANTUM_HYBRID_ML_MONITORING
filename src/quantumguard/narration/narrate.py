"""Plain-language narration of trend decisions.

Narration is strictly downstream of the decision path: it renders an already
frozen TrendDecision and never influences it (CLAUDE.md). The default is a
deterministic template; an LLM rendering can be enabled via config but is off
by default and raises if invoked without being wired up.
"""

from __future__ import annotations

from quantumguard.config import Config, load_config
from quantumguard.health.mhs import HealthComponents
from quantumguard.health.trend import TrendDecision

_NOT_TRIGGERED_TEXT = {
    "insufficient_history": "Not enough MHS history yet to evaluate a trend.",
    "mhs_healthy": "The model health score is still in the healthy band, so no retraining is considered.",
    "no_signal": "Neither the rolling slope nor the CUSUM change-point shows a significant decline.",
}


def narrate(decision: TrendDecision, components: HealthComponents | None = None) -> str:
    parts: list[str] = []

    current = decision.series[-1] if decision.series else float("nan")
    parts.append(f"At window {decision.window_id}, the model health score is {current:.2f}.")

    if components is not None:
        parts.append(
            f"Status is {components.status}: classical drift {components.cds:.2f}, "
            f"quantum drift {components.qds:.2f}, prediction confidence {components.pc:.2f}, "
            f"accuracy trend {components.at:.2f}."
        )
        if components.cds_fraud is not None and components.cds_fraud > components.cds_population:
            parts.append(
                "Drift is concentrated in the fraud-labeled subset "
                f"(fraud {components.cds_fraud:.2f} vs population {components.cds_population:.2f})."
            )

    if decision.triggered:
        slope_text = (
            f"the health trend slope is {decision.slope:.4f} per window (p={decision.p_value:.3g})"
        )
        cusum_text = (
            f"the CUSUM statistic reached {decision.cusum_stat:.1f} "
            f"(threshold {decision.cusum_threshold:.1f})"
        )
        evidence = {
            "slope": slope_text,
            "cusum": cusum_text,
            "slope+cusum": f"{slope_text} and {cusum_text}",
        }[decision.trigger_reason]
        width_desc = "narrow" if decision.drift_type == "sudden" else "wide"
        parts.append(
            f"Retraining was triggered because {evidence}. The decline looks "
            f"{decision.drift_type}, so a {width_desc} retraining window of "
            f"{decision.retrain_window_width} windows of recent data will be used."
        )
    else:
        parts.append(_NOT_TRIGGERED_TEXT[decision.not_triggered_reason])

    return " ".join(parts)


LLM_SYSTEM_PROMPT = (
    "You narrate decisions made by QuantumGuard, an automated fraud-model "
    "drift-monitoring system, for an operations audience. You will receive the "
    "complete, already-final decision record. Restate it in clear plain English "
    "in one short paragraph. You must not second-guess, extend, or alter the "
    "decision, and you must not introduce any number that is not in the record."
)


def build_llm_prompt(decision: TrendDecision, components: HealthComponents | None = None) -> str:
    """The user-turn prompt for LLM narration: the frozen decision facts plus
    the deterministic template rendering as grounding. Pure and testable."""
    facts = [
        f"window_id: {decision.window_id}",
        f"current_mhs: {decision.series[-1]:.4f}" if decision.series else "current_mhs: n/a",
        f"slope: {decision.slope:.6f} (p={decision.p_value:.4g})",
        f"cusum_stat: {decision.cusum_stat:.2f} (threshold {decision.cusum_threshold:.2f})",
        f"triggered: {decision.triggered}",
        f"trigger_reason: {decision.trigger_reason}",
        f"not_triggered_reason: {decision.not_triggered_reason}",
        f"drift_type: {decision.drift_type}",
        f"retrain_window_width: {decision.retrain_window_width}",
    ]
    if components is not None:
        facts += [
            f"status: {components.status}",
            f"cds: {components.cds:.4f} (population {components.cds_population:.4f}, "
            f"fraud {components.cds_fraud})",
            f"qds: {components.qds:.4f}",
            f"pc: {components.pc:.4f}",
            f"at: {components.at:.4f}",
        ]
    return (
        "Decision record:\n" + "\n".join(facts)
        + "\n\nDeterministic template rendering (for reference):\n"
        + narrate(decision, components)
    )


def narrate_llm(decision: TrendDecision, components: HealthComponents | None = None,
                config: Config | None = None) -> str:
    """LLM rendering of an already-frozen decision. Off by default; the
    templated narrate() is always the canonical fallback."""
    cfg = config if config is not None else load_config()
    if not cfg.narration.llm_enabled:
        raise RuntimeError("LLM narration is disabled (narration.llm_enabled=false); use narrate()")

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "LLM narration requires the anthropic package: uv sync --extra llm"
        ) from exc

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=cfg.narration.llm_model,
        max_tokens=cfg.narration.llm_max_tokens,
        system=LLM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_llm_prompt(decision, components)}],
    )
    if response.stop_reason == "refusal":
        return narrate(decision, components)  # canonical template as fallback
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    return text if text else narrate(decision, components)
