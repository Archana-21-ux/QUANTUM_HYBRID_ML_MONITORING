import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";

// ---- helpers ---------------------------------------------------------------

function parseTs(ts) {
  if (!ts) return null;
  // health events carry ISO ts (with offset); SQLite rows are "YYYY-MM-DD HH:MM:SS" (UTC)
  const iso = ts.includes("T") ? ts : ts.replace(" ", "T") + "Z";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

function fmtClock(ts) {
  const d = parseTs(ts);
  if (!d) return "--:--:--";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function fmtDuration(sec) {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return "--:--";
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

// The engine emits 3 status bands (healthy/warning/critical). We surface 4
// recovery states by splitting "warning" into Degraded vs Recovering based on
// whether MHS is trending up — i.e. actually climbing back toward healthy.
function currentState(health) {
  if (!health.length) return { key: "unknown", label: "Unknown", emoji: "⚪" };
  const last = health[health.length - 1];
  if (last.status === "critical") return { key: "critical", label: "Critical", emoji: "🔴" };
  if (last.status === "healthy") return { key: "healthy", label: "Healthy", emoji: "🟢" };
  const prev = health[Math.max(0, health.length - 4)];
  return last.mhs > prev.mhs
    ? { key: "recovering", label: "Recovering", emoji: "🟡" }
    : { key: "degraded", label: "Degraded", emoji: "🟠" };
}

// Segment the health series into degradation dips (contiguous non-healthy runs).
// A dip opens when the model leaves Healthy and records the WORST state it
// reached (Critical if MHS<0.5, else Degraded), plus the timestamps of each
// state on the way. A dip is "recovered" when a new model is DEPLOYED after its
// onset (the fix), or — as a fallback — when health organically returns to
// Healthy. Recovery Time = recovery timestamp − onset (onset = the Critical
// timestamp when the dip went critical, else the moment it left Healthy). The
// timer resets on each fresh dip.
function segmentDips(health) {
  const segs = [];
  let s = null;
  for (const w of health) {
    if (w.status !== "healthy") {
      if (!s) s = { startAt: w.ts, criticalAt: null, degradedAt: null, recoveringAt: null,
                    healthyAt: null, worst: "degraded" };
      if (w.status === "critical") {
        if (!s.criticalAt) s.criticalAt = w.ts;
        s.worst = "critical";
      } else {
        if (!s.degradedAt) s.degradedAt = w.ts;
        s.recoveringAt = w.ts;
      }
    } else if (s) {
      s.healthyAt = w.ts;                 // organic return to Healthy (fallback marker)
      segs.push(s);
      s = null;
    }
  }
  if (s) segs.push(s);                    // trailing (still non-healthy) dip
  return segs;
}

function analyseRecovery(health, deployments, nowMs) {
  const ms = (t) => {
    const d = parseTs(t);
    return d ? d.getTime() : null;
  };
  const firstHealthyAfter = (afterMs) => {
    for (const w of health) if (w.status === "healthy" && ms(w.ts) > afterMs) return w.ts;
    return null;
  };
  // states reached between an anchor time and an optional end time
  const timelineSince = (anchorMs, endMs) => {
    const tl = { criticalAt: null, degradedAt: null, recoveringAt: null, healthyAt: null, worst: "degraded" };
    for (const w of health) {
      const t = ms(w.ts);
      if (t == null || t < anchorMs || (endMs != null && t > endMs)) continue;
      if (w.status === "critical") { if (!tl.criticalAt) tl.criticalAt = w.ts; tl.worst = "critical"; }
      else if (w.status === "warning") { if (!tl.degradedAt) tl.degradedAt = w.ts; tl.recoveringAt = w.ts; }
      else if (w.status === "healthy") { if (!tl.healthyAt) tl.healthyAt = w.ts; }
    }
    return tl;
  };

  // completed recovery per deployment: deploy -> first Healthy before the next deploy
  const done = [];
  deployments.forEach((d, i) => {
    const dMs = ms(d.ts);
    const nextMs = i + 1 < deployments.length ? ms(deployments[i + 1].ts) : Infinity;
    for (const w of health) {
      const t = ms(w.ts);
      if (w.status === "healthy" && t > dMs && t < nextMs) { done.push((t - dMs) / 1000); break; }
    }
  });
  const avg = done.length ? done.reduce((a, b) => a + b, 0) / done.length : null;
  const fastest = done.length ? Math.min(...done) : null;

  // The recovery clock is anchored at the LATEST deployment, so it RESTARTS
  // FROM ZERO every time a new model is deployed and ticks up until the newly
  // deployed model brings health back to Healthy. Before any deployment it
  // falls back to the current degradation onset.
  let anchorMs = null;
  if (deployments.length) {
    anchorMs = ms(deployments[deployments.length - 1].ts);
  } else {
    const segs = segmentDips(health);
    if (segs.length) {
      const s = segs[segs.length - 1];
      anchorMs = ms(s.criticalAt || s.startAt);
    }
  }

  if (anchorMs == null) {
    return { ttrSeconds: null, ongoing: false, worstLabel: "Critical",
             timeline: { criticalAt: null, degradedAt: null, recoveringAt: null, healthyAt: null },
             avg, fastest, count: done.length };
  }

  const healthyTs = firstHealthyAfter(anchorMs);
  const ongoing = healthyTs == null;                       // still recovering since the deploy
  const ttrSeconds = ongoing ? (nowMs - anchorMs) / 1000 : (ms(healthyTs) - anchorMs) / 1000;
  const timeline = timelineSince(anchorMs, healthyTs ? ms(healthyTs) : null);
  const worstLabel = timeline.worst === "critical" ? "Critical" : "Degraded";

  return { ttrSeconds, ongoing, worstLabel, timeline, avg, fastest, count: done.length };
}

const JOURNEY = [
  { key: "critical", label: "Critical", emoji: "🔴", ts: (t) => t?.criticalAt },
  { key: "degraded", label: "Degraded", emoji: "🟠", ts: (t) => t?.degradedAt },
  { key: "recovering", label: "Recovering", emoji: "🟡", ts: (t) => t?.recoveringAt },
  { key: "healthy", label: "Healthy", emoji: "🟢", ts: (t) => t?.healthyAt },
];

function Metric({ label, value }) {
  return (
    <div className="rec-metric">
      <div className="rec-metric-val">{value}</div>
      <div className="rec-metric-label">{label}</div>
    </div>
  );
}

function InfoRow({ label, value, tone }) {
  return (
    <div className="rec-info-row">
      <span className="rec-info-label">{label}</span>
      <span className={`rec-info-val ${tone || ""}`}>{value}</span>
    </div>
  );
}

// ---- component -------------------------------------------------------------

export default function ModelRecovery({ health = [], deployments = [], refreshSignal }) {
  const [now, setNow] = useState(Date.now());
  const [meta, setMeta] = useState({ f1: null, lastRetrain: null, retrainStatus: "Idle" });

  // live tick drives the ongoing recovery timer (the card is always visible)
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // F1 + last-retraining come from the same candidates endpoint the retrain panel uses
  useEffect(() => {
    let alive = true;
    api
      .candidates()
      .then((body) => {
        if (!alive) return;
        const cands = body.candidates || [];
        const newest = cands[0]; // API returns most-recent first
        const f1 = newest?.offline_metrics?.deployed?.fraud_f1 ?? null;
        const lastRetrain = newest ? newest.decided_at || newest.created_at : null;
        const retrainStatus = cands.some((c) => c.status === "pending")
          ? "In progress"
          : cands.length
          ? "Completed"
          : "Idle";
        setMeta({ f1, lastRetrain, retrainStatus });
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [refreshSignal]);

  const state = useMemo(() => currentState(health), [health]);
  const rec = useMemo(() => analyseRecovery(health, deployments, now), [health, deployments, now]);
  const last = health[health.length - 1];
  const healthScore = last ? Math.round(last.mhs * 100) : null;
  const driftScore = last ? Math.max(last.cds ?? 0, last.qds ?? 0) : null;
  const driftDetected = driftScore != null && driftScore >= 0.2;

  // horizontal card in the main layout, directly below the health chart and
  // drift breakdown: status · time to recovery · journey · stats · monitoring
  return (
    <section className="recovery-row">
      <div className={`card recovery-card state-${state.key}`} aria-label="Model Recovery">
        <div className="card-head">
          <span className="card-title">Model Recovery</span>
          <span className="rec-head-sub">Real-time model health &amp; recovery</span>
        </div>

        <div className="rec-grid">
          <div className={`rec-status-row state-${state.key}`}>
            <div className="rec-status">
              <span className="rec-status-dot" />
              <div>
                <div className="rec-status-label">{state.emoji} {state.label}</div>
                <div className="rec-status-sub">Current status</div>
              </div>
            </div>
            <div className="rec-score">
              <div className="rec-score-val">{healthScore ?? "--"}<span>/100</span></div>
              <div className="rec-score-label">Health score</div>
            </div>
          </div>

          <div className={`rec-ttr state-${state.key} ${rec.ongoing ? "live" : ""}`}>
            <div className="rec-ttr-label">Time to Recovery{rec.ongoing ? " · live" : ""}</div>
            <div className="rec-ttr-value">{fmtDuration(rec.ttrSeconds)}</div>
            <div className="rec-ttr-sub">{rec.worstLabel} → Healthy</div>
          </div>

          <div className="rec-col rec-col-journey">
            <div className="rec-section-label">Recovery Journey</div>
            <div className="rec-journey">
              {JOURNEY.map((step, i) => {
                const ts = step.ts(rec.timeline);
                const reached = !!ts;
                return (
                  <React.Fragment key={step.key}>
                    <div className={`rec-step ${reached ? "reached" : ""} state-${step.key}`}>
                      <div className="rec-step-dot">{step.emoji}</div>
                      <div className="rec-step-name">{step.label}</div>
                      <div className="rec-step-ts">{reached ? fmtClock(ts) : "--:--"}</div>
                    </div>
                    {i < JOURNEY.length - 1 && <div className={`rec-step-line ${reached ? "reached" : ""}`} />}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          <div className="rec-col">
            <div className="rec-section-label">Recovery Stats</div>
            <div className="rec-metrics">
              <Metric label="Avg recovery" value={fmtDuration(rec.avg)} />
              <Metric label="Fastest" value={fmtDuration(rec.fastest)} />
              <Metric label="Events" value={rec.count} />
            </div>
          </div>

          <div className="rec-col">
            <div className="rec-section-label">Monitoring</div>
            <div className="rec-info">
              <InfoRow label="Drift status" value={driftDetected ? "Detected" : "Stable"} tone={driftDetected ? "warn" : "ok"} />
              <InfoRow label="Drift score" value={driftScore != null ? driftScore.toFixed(2) : "--"} />
              <InfoRow label="F1 score" value={meta.f1 != null ? meta.f1.toFixed(2) : "--"} tone="ok" />
              <InfoRow label="Last retraining" value={meta.lastRetrain ? fmtClock(meta.lastRetrain) : "--:--"} />
              <InfoRow
                label="Retraining status"
                value={meta.retrainStatus}
                tone={meta.retrainStatus === "Completed" ? "ok" : meta.retrainStatus === "In progress" ? "warn" : ""}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
