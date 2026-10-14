"""Sliding-window model comparison: pick the best performer among the new
candidate, the deployed model, and retained past deployed versions, all
evaluated on the same recent-regime holdout."""

from __future__ import annotations

from typing import Any


def pick_winner(
    scores: dict[str, dict[str, Any]], *, metric: str, tiebreak: str
) -> str:
    """Version with the highest `metric` (ties broken by `tiebreak`)."""
    if not scores:
        raise ValueError("no scores to compare")
    return max(scores, key=lambda version: (scores[version][metric], scores[version][tiebreak]))


def comparison_log_line(
    *,
    window_id: int,
    candidate_version: str,
    deployed_version: str,
    past_versions: list[str],
    scores: dict[str, dict[str, Any]],
    winner: str,
    metric: str,
) -> str:
    """Plain-text evaluation record for the reasoning log, e.g.
    'Evaluated v9(new), v8(deployed), v7, v6 on window 300 — v6 scored
    highest (fraud_f1=0.981) — rolling back to v6.'"""
    contenders = [f"{candidate_version}(new)", f"{deployed_version}(deployed)"] + list(past_versions)
    if winner == candidate_version:
        winner_label, action = f"{winner}(new)", "candidate pending approval."
    elif winner == deployed_version:
        winner_label, action = f"{winner}(deployed)", "keeping deployed model, candidate rejected."
    else:
        winner_label, action = winner, f"rolling back to {winner}."
    return (
        f"Evaluated {', '.join(contenders)} on window {window_id} — "
        f"{winner_label} scored highest ({metric}={scores[winner][metric]:.3f}) — {action}"
    )
