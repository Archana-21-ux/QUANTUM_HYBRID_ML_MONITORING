import React from "react";

const KIND_ICONS = {
  info: "·",
  status: "◆",
  trigger: "⚠",
  candidate: "★",
  deploy: "▲",
};

export default function ReasoningLog({ entries, highlight }) {
  return (
    <div className={`card ${highlight ? "highlight" : ""}`}>
      <div className="card-head">
        <span className="card-title">Agent Reasoning Log</span>
        <span className="kpi-icon">☰</span>
      </div>
      <div className="scroll-body">
        {entries.length === 0 && <p className="empty">decisions will be narrated here</p>}
        {entries.map((entry, index) => (
          <div key={index} className={`log-entry ${entry.kind}`}>
            <span className="log-icon">{KIND_ICONS[entry.kind] ?? "·"}</span>
            <span className="log-ts">
              {entry.ts?.slice(11, 19)}
            </span>
            <span>{entry.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
