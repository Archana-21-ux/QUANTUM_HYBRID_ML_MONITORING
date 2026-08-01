import React from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const AXIS = { stroke: "#a9b5ac", fontSize: 11 };

function HealthChart({ health }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={health} margin={{ top: 8, right: 10, bottom: 0, left: -22 }}>
        <defs>
          <linearGradient id="mhsFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#16a06a" stopOpacity={0.28} />
            <stop offset="100%" stopColor="#16a06a" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 4" stroke="#eef1ed" vertical={false} />
        <XAxis dataKey="window_id" type="number" domain={["dataMin", "dataMax"]} tick={AXIS} />
        <YAxis domain={[0, 1]} tick={AXIS} />
        <Tooltip contentStyle={{ borderRadius: 10, border: "1px solid #e3e8e2", fontSize: 12 }} />
        <ReferenceLine y={0.75} stroke="#16a06a" strokeDasharray="4 4" strokeOpacity={0.5} />
        <ReferenceLine y={0.5} stroke="#d9483b" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Area dataKey="mhs" stroke="#16a06a" strokeWidth={2.4} fill="url(#mhsFill)"
              isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function DriftChart({ health }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={health} margin={{ top: 8, right: 10, bottom: 0, left: -22 }}>
        <CartesianGrid strokeDasharray="3 4" stroke="#eef1ed" vertical={false} />
        <XAxis dataKey="window_id" type="number" domain={["dataMin", "dataMax"]} tick={AXIS} />
        <YAxis domain={[0, 1]} tick={AXIS} />
        <Tooltip contentStyle={{ borderRadius: 10, border: "1px solid #e3e8e2", fontSize: 12 }} />
        <Line dataKey="cds_population" stroke="#e06c3c" strokeWidth={2} dot={false}
              isAnimationActive={false} />
        <Line dataKey="cds_fraud" stroke="#e06c3c" strokeWidth={1.4} strokeDasharray="5 4"
              dot={false} connectNulls={false} isAnimationActive={false} />
        <Line dataKey="qds_population" stroke="#3ca7e0" strokeWidth={2} dot={false}
              isAnimationActive={false} />
        <Line dataKey="qds_fraud" stroke="#3ca7e0" strokeWidth={1.4} strokeDasharray="5 4"
              dot={false} connectNulls={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function MainChart({ health, tab, onTab }) {
  return (
    <div className="card chart-card">
      <div className="card-head">
        <span className="card-title">
          {tab === "drift" ? "Drift Signals — population vs fraud subset" : "Model Health Score"}
        </span>
        <div className="tabbar">
          <button className={tab !== "drift" ? "active" : ""} onClick={() => onTab("health")}>
            Health
          </button>
          <button className={tab === "drift" ? "active" : ""} onClick={() => onTab("drift")}>
            Drift
          </button>
        </div>
      </div>
      <div className="chart-area">
        {tab === "drift" ? <DriftChart health={health} /> : <HealthChart health={health} />}
      </div>
      <div className="chart-legend">
        {tab === "drift" ? (
          <>
            <span style={{ color: "#e06c3c" }}>■ <b>classical (CDS)</b></span>
            <span style={{ color: "#3ca7e0" }}>■ <b>quantum (QDS)</b></span>
            <span>solid = population · dashed = fraud subset</span>
          </>
        ) : (
          <>
            <span style={{ color: "#16a06a" }}>■ <b>MHS</b></span>
            <span>dashed lines = healthy / critical bands</span>
          </>
        )}
      </div>
    </div>
  );
}
