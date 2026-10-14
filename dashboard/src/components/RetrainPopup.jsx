import React, { useEffect, useState } from "react";
import { api } from "../api.js";

// Small decision pop-up shown whenever a retraining candidate is pending.
// It is additional to the Model Retraining panel (which stays unchanged) and
// uses the same approve/reject endpoints. "Later" snoozes the current
// candidate; the pop-up returns when a different candidate becomes pending.

const METRICS = [
  ["fraud_precision", "Precision"],
  ["fraud_recall", "Recall"],
  ["fraud_f1", "F1-score"],
  ["roc_auc", "AUC"],
];

function fmtTime(ts) {
  if (!ts) return "--";
  const d = new Date(ts.includes("T") ? ts : ts.replace(" ", "T") + "Z");
  return Number.isNaN(d.getTime())
    ? ts
    : d.toLocaleString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, day: "2-digit", month: "short" });
}

function Detail({ label, value }) {
  return (
    <div className="rp-detail">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function RetrainPopup({ refreshSignal, onDecided }) {
  const [candidate, setCandidate] = useState(null);
  const [snoozed, setSnoozed] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function refresh() {
    const body = await api.candidates();
    const pending = (body.candidates || []).filter((c) => c.status === "pending");
    setCandidate(pending[0] || null); // API returns most-recent first
  }

  useEffect(() => {
    refresh().catch(() => {});
  }, [refreshSignal]);

  const open = candidate && !snoozed.has(candidate.id);

  function later() {
    setSnoozed((s) => new Set(s).add(candidate.id));
  }

  // Esc behaves like "Later"
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && !busy && later();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, busy, candidate]);

  async function decide(action) {
    setBusy(true);
    setError(null);
    try {
      await (action === "approve" ? api.approve(candidate.id) : api.reject(candidate.id));
      await refresh();
      onDecided?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;

  const m = candidate.offline_metrics || {};
  const isRollback = !!m.rollback;
  const blend = m.blend;
  const compared = m.comparison ? Object.keys(m.comparison).length : 0;
  const shadow = candidate.shadow;

  return (
    <div className="rp-backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && later()}>
      <div className="rp-modal" role="dialog" aria-modal="true" aria-labelledby="rp-title">
        <div className="rp-head">
          <span className="rp-icon">⟳</span>
          <div>
            <h3 id="rp-title">Model retraining ready for review</h3>
            <p>
              {isRollback ? "Rollback to" : "Candidate"} <strong>{candidate.model_version}</strong>
              {m.deployed_version && <> vs deployed <strong>{m.deployed_version}</strong></>}
            </p>
          </div>
          <span className={`tag ${isRollback ? "rollback" : "pending"}`}>{isRollback ? "rollback" : "pending"}</span>
        </div>

        {m.candidate && m.deployed && (
          <table className="metrics rp-metrics">
            <thead>
              <tr>
                <th>fraud metric</th>
                <th>deployed</th>
                <th>{isRollback ? "rollback" : "candidate"}</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map(([key, label]) => {
                const delta = m.deltas?.[key] ?? m.candidate[key] - m.deployed[key];
                return (
                  <tr key={key}>
                    <td>{label}</td>
                    <td>{m.deployed[key]?.toFixed(3)}</td>
                    <td>{m.candidate[key]?.toFixed(3)}</td>
                    <td className={delta >= 0 ? "pos" : "neg"}>
                      {delta >= 0 ? "+" : ""}
                      {delta.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        <div className="rp-details">
          <Detail label="Created" value={fmtTime(candidate.created_at)} />
          <Detail
            label="Evaluation set"
            value={m.candidate?.n_test ? `${m.candidate.n_test} txns · ${m.candidate.n_test_fraud} fraud` : m.eval_set || "--"}
          />
          {blend && (
            <Detail
              label="Training data"
              value={`${blend.n_recent} recent + ${blend.n_historical} historical`}
            />
          )}
          {blend?.recent_windows && (
            <Detail label="Drift windows" value={`${blend.recent_windows[0]} – ${blend.recent_windows[1]}`} />
          )}
          {compared > 0 && <Detail label="Compared against" value={`${compared} past versions`} />}
          <Detail
            label="Shadow validation"
            value={shadow ? `${(shadow.overall_agreement * 100).toFixed(1)}% agree · ${shadow.n_windows} windows` : "in progress"}
          />
        </div>

        {error && <p className="rp-error">{error}</p>}

        <div className="rp-actions">
          <button className="rp-later" disabled={busy} onClick={later}>Later</button>
          <button className="rp-reject" disabled={busy} onClick={() => decide("reject")}>Reject</button>
          <button className="rp-approve" disabled={busy} onClick={() => decide("approve")}>
            {busy ? "Working…" : "Approve & deploy"}
          </button>
        </div>
      </div>
    </div>
  );
}
