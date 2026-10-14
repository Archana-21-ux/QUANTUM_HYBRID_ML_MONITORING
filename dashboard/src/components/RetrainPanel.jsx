import React, { useEffect, useState } from "react";
import { api } from "../api.js";

const METRIC_LABELS = {
  fraud_precision: "precision",
  fraud_recall: "recall",
  fraud_f1: "F1",
  roc_auc: "AUC",
};

function Candidate({ candidate, busy, onDecide }) {
  const metrics = candidate.offline_metrics;
  // a pending rollback proposal (past version won the sliding comparison)
  // reuses the badge slot with a distinct amber ROLLBACK tag — no new UI
  const isRollback = candidate.status === "pending" && metrics?.rollback;
  return (
    <div>
      <div className="cand-head">
        <strong>{candidate.model_version}</strong>
        <span className={`tag ${isRollback ? "rollback" : candidate.status}`}>
          {isRollback ? "rollback" : candidate.status}
        </span>
      </div>
      {candidate.shadow && (
        <div className="cand-shadow">
          shadow agreement {(candidate.shadow.overall_agreement * 100).toFixed(1)}% over{" "}
          {candidate.shadow.n_windows} windows
        </div>
      )}
      {metrics && (
        <table className="metrics">
          <thead>
            <tr>
              <th>fraud metric</th>
              <th>deployed</th>
              <th>{metrics.rollback ? "rollback" : "candidate"}</th>
              <th>Δ</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(METRIC_LABELS).map(([key, label]) => (
              <tr key={key}>
                <td>{label}</td>
                <td>{metrics.deployed[key].toFixed(3)}</td>
                <td>{metrics.candidate[key].toFixed(3)}</td>
                <td className={metrics.deltas[key] >= 0 ? "pos" : "neg"}>
                  {metrics.deltas[key] >= 0 ? "+" : ""}
                  {metrics.deltas[key].toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {candidate.status === "pending" ? (
        <div className="actions">
          <button className="approve" disabled={busy} onClick={() => onDecide("approve")}>
            Approve &amp; deploy
          </button>
          <button className="reject" disabled={busy} onClick={() => onDecide("reject")}>
            Reject
          </button>
        </div>
      ) : (
        candidate.decided_by && (
          <p className="decided-note">
            {candidate.status} by {candidate.decided_by} at {candidate.decided_at}
          </p>
        )
      )}
    </div>
  );
}

export default function RetrainPanel({ refreshSignal, highlight }) {
  const [candidates, setCandidates] = useState([]);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    const body = await api.candidates();
    setCandidates(body.candidates);
  }

  useEffect(() => {
    refresh().catch(console.error);
  }, [refreshSignal]);

  async function decide(id, action) {
    setBusy(true);
    try {
      await (action === "approve" ? api.approve(id) : api.reject(id));
      await refresh();
    } catch (error) {
      alert(error.message);
    } finally {
      setBusy(false);
    }
  }

  const pending = candidates.filter((c) => c.status === "pending");
  const shown = pending.length ? pending : candidates.slice(0, 1);

  return (
    <div className={`card ${highlight ? "highlight" : ""}`}>
      <div className="card-head">
        <span className="card-title">
          Model Retraining{pending.length > 0 && ` — ${pending.length} pending`}
        </span>
        <span className="kpi-icon">⟳</span>
      </div>
      <div className="scroll-body">
        {shown.length === 0 ? (
          <p className="cand-empty">
            no retraining candidates yet — they appear when the trend detector triggers
          </p>
        ) : (
          shown.map((candidate) => (
            <Candidate
              key={candidate.id}
              candidate={candidate}
              busy={busy}
              onDecide={(action) => decide(candidate.id, action)}
            />
          ))
        )}
      </div>
    </div>
  );
}
