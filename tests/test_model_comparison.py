import pytest

from quantumguard.retrain.comparison import comparison_log_line, pick_winner

SCORES = {
    "v9": {"fraud_f1": 0.90, "roc_auc": 0.99},   # new candidate
    "v8": {"fraud_f1": 0.85, "roc_auc": 0.99},   # deployed
    "v7": {"fraud_f1": 0.90, "roc_auc": 0.995},  # past — ties candidate on f1, wins tiebreak
    "v6": {"fraud_f1": 0.60, "roc_auc": 0.90},
}


def test_pick_winner_uses_metric_then_tiebreak():
    assert pick_winner(SCORES, metric="fraud_f1", tiebreak="roc_auc") == "v7"
    no_tie = {**SCORES, "v7": {"fraud_f1": 0.80, "roc_auc": 1.0}}
    assert pick_winner(no_tie, metric="fraud_f1", tiebreak="roc_auc") == "v9"
    with pytest.raises(ValueError):
        pick_winner({}, metric="fraud_f1", tiebreak="roc_auc")


def test_log_line_candidate_wins():
    scores = {**SCORES, "v7": {"fraud_f1": 0.80, "roc_auc": 1.0}}
    line = comparison_log_line(
        window_id=300, candidate_version="v9", deployed_version="v8",
        past_versions=["v7", "v6"], scores=scores, winner="v9", metric="fraud_f1",
    )
    assert line == (
        "Evaluated v9(new), v8(deployed), v7, v6 on window 300 — "
        "v9(new) scored highest (fraud_f1=0.900) — candidate pending approval."
    )


def test_log_line_rollback():
    line = comparison_log_line(
        window_id=300, candidate_version="v9", deployed_version="v8",
        past_versions=["v7", "v6"], scores=SCORES, winner="v7", metric="fraud_f1",
    )
    assert "v7 scored highest" in line
    assert line.endswith("rolling back to v7.")


def test_log_line_deployed_wins():
    scores = {v: {"fraud_f1": 0.5, "roc_auc": 0.9} for v in ("v9", "v7")}
    scores["v8"] = {"fraud_f1": 0.95, "roc_auc": 0.99}
    line = comparison_log_line(
        window_id=42, candidate_version="v9", deployed_version="v8",
        past_versions=["v7"], scores=scores, winner="v8", metric="fraud_f1",
    )
    assert "v8(deployed) scored highest" in line
    assert line.endswith("keeping deployed model, candidate rejected.")
