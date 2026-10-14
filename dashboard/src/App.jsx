import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import { useLiveFeed } from "./useLiveFeed.js";
import MainChart from "./components/MainChart.jsx";
import DriftBreakdown from "./components/DriftBreakdown.jsx";
import PredictionsPanel from "./components/PredictionsPanel.jsx";
import RetrainPanel from "./components/RetrainPanel.jsx";
import ReasoningLog from "./components/ReasoningLog.jsx";
import ModelRecovery from "./components/ModelRecovery.jsx";
import RetrainPopup from "./components/RetrainPopup.jsx";

function Kpi({ label, icon, value, delta, note, invert = false, badge = null }) {
  const hasDelta = delta !== null && !Number.isNaN(delta);
  const good = invert ? delta <= 0 : delta >= 0;
  return (
    <div className="card kpi">
      <div className="kpi-top">
        <span className="card-title">{label}</span>
        <span className="kpi-icon">{icon}</span>
      </div>
      <div className="kpi-value-row">
        <span className="kpi-value">{value}</span>
        {badge && <span className="pill ver" title="deployed model version">{badge}</span>}
        {hasDelta && (
          <span className={`pill ${good ? "pos" : "neg"}`}>
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(3)}
          </span>
        )}
      </div>
      <span className="kpi-note">{note}</span>
    </div>
  );
}

const LIVE_OPTION = "__live__";

export default function App() {
  const feed = useLiveFeed();
  const [scenarios, setScenarios] = useState([]);
  const [scenario, setScenario] = useState("");
  const [busy, setBusy] = useState(false);
  const [section, setSection] = useState("dashboard");
  const [decisionBump, setDecisionBump] = useState(0); // pop-up decisions -> panels refetch
  const [simState, setSimState] = useState(null); // server-side sim state (authoritative)

  // initial load: scenarios + starting selection + current server state
  useEffect(() => {
    api
      .simulationStatus()
      .then((body) => {
        setScenarios(body.scenarios);
        setSimState(body.status.state);
        setScenario((s) => s || body.scenarios[0] || "");
      })
      .catch(() => {});
  }, []);

  // keep the Start/Stop button in sync with the SERVER even if the WebSocket
  // drops, so it never offers "Start" while a run is already active
  useEffect(() => {
    const id = setInterval(() => {
      api.simulationStatus().then((body) => setSimState(body.status.state)).catch(() => {});
    }, 4000);
    return () => clearInterval(id);
  }, []);

  const latest = feed.health[feed.health.length - 1];
  const previous = feed.health[feed.health.length - 2];
  const running = feed.simulation.state === "running" || simState === "running";
  const delta = (key) => (latest && previous ? latest[key] - previous[key] : null);
  const kpiValue = (key, digits = 2) => (latest ? latest[key].toFixed(digits) : "—");
  const windowNote = latest ? `window ${latest.window_id}` : "waiting for simulation";

  async function toggleSimulation() {
    setBusy(true);
    try {
      if (running) {
        await api.stopSimulation();
        setSimState("idle");
      } else {
        await api.startSimulation(scenario);
        setSimState("running");
      }
    } catch (error) {
      // a run is already active on the server — reflect it instead of erroring
      if (String(error.message).includes("already running")) setSimState("running");
      else alert(error.message);
    } finally {
      setBusy(false);
    }
  }

  function onScenarioChange(event) {
    setScenario(event.target.value);
  }

  const chartTab = section === "drift" ? "drift" : "health";

  return (
    <div className="shell">
      <div className={`main${chartTab === "drift" ? " view-drift" : ""}`}>
        <header className="topbar">
          <div>
            <h1>ML MODEL MONITORING</h1>
            <div className="subtitle">quantum classical hybrid drift monitoring &amp; semi - retraining</div>
          </div>
          <div className="controls">
            <span className="feed-status" title={feed.connected ? "receiving live updates" : "reconnecting to the server"}>
              <span className={feed.connected ? "dot on" : "dot off"} />
              {feed.connected ? "live feed connected" : "reconnecting…"}
            </span>
            <span className="header-deployed" title="currently deployed model version">
              deployed <strong>{feed.deployed ?? "…"}</strong>
            </span>
            {latest && (
              <span className={`status-pill ${latest.status}`}>
                {latest.status.toUpperCase()}
              </span>
            )}
            <select value={scenario} onChange={onScenarioChange} disabled={running}>
              {scenarios.map((name) => (
                <option key={name}>{name}</option>
              ))}
              <option value={LIVE_OPTION}>⚡ Live Streaming Data</option>
            </select>
            <button className="primary" onClick={toggleSimulation} disabled={busy || !scenario}>
              {running ? "Stop" : "Start simulation"}
            </button>
          </div>
        </header>

        <section className="kpis">
          <Kpi label="Model Health" icon="♥" value={kpiValue("mhs")}
               delta={delta("mhs")} note={windowNote} badge={feed.deployed} />
          <Kpi label="Classical Drift" icon="∿" value={kpiValue("cds")}
               delta={delta("cds")} note="fused CDS" invert />
          <Kpi label="Quantum Drift" icon="◬" value={kpiValue("qds")}
               delta={delta("qds")} note="fused QDS" invert />
          <Kpi label="Accuracy Trend" icon="✓" value={kpiValue("at")}
               delta={delta("at")} note="lagged-label AUC ratio" />
        </section>

        <section className="middle">
          <MainChart
            health={feed.health}
            tab={chartTab}
            onTab={(tab) => setSection(tab === "drift" ? "drift" : "dashboard")}
          />
          <DriftBreakdown
            latest={latest}
            featureDrift={feed.featureDrift}
            driftView={chartTab === "drift"}
          />
        </section>

        {/* Model Recovery: full-width horizontal row under the chart + drift breakdown */}
        <ModelRecovery health={feed.health} deployments={feed.deployments} refreshSignal={feed.candidateEvents} />

        <section className="bottom">
          <RetrainPanel refreshSignal={feed.candidateEvents + decisionBump} />
          <PredictionsPanel predictions={feed.predictions} />
        </section>

        {/* Agent Reasoning Log gets its own full-width row so decisions
            (drift triggers, cooldowns, candidates, approvals, rollbacks) stay
            prominent and readable during a live walkthrough */}
        <ReasoningLog entries={feed.reasoning} />
      </div>

      {/* additional retraining decision pop-up (the retraining panel is unchanged) */}
      <RetrainPopup
        refreshSignal={feed.candidateEvents + decisionBump}
        onDecided={() => setDecisionBump((n) => n + 1)}
      />
    </div>
  );
}
