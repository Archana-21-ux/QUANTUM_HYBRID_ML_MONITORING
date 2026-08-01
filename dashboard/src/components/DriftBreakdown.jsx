import React, { useMemo, useState } from "react";

function Row({ name, value, sub }) {
  const width = value === null ? 0 : Math.min(value * 100, 100);
  const tone = value === null ? "" : value >= 0.5 ? "crit" : value >= 0.25 ? "warn" : "";
  return (
    <div className="breakdown-row">
      <div className="breakdown-top">
        <span className="breakdown-name">{name}</span>
        <span className="breakdown-value">{value === null ? "—" : value.toFixed(3)}</span>
      </div>
      <div className="bar">
        <div className={`bar-fill ${tone}`} style={{ width: `${width}%` }} />
      </div>
      <span className="breakdown-sub">{sub}</span>
    </div>
  );
}

const COLUMNS = [
  { key: "feature", label: "Feature" },
  { key: "classical_drift", label: "Classical" },
  { key: "quantum_drift", label: "Quantum" },
  { key: "hybrid_drift", label: "Hybrid" },
  { key: "severity", label: "Status" },
];
const SEVERITY_RANK = { ok: 0, warn: 1, crit: 2 };

function DriftCell({ value }) {
  // severity color bands: green < 0.2, amber 0.2-0.5, red > 0.5
  const tone = value >= 0.5 ? "crit" : value >= 0.2 ? "warn" : "ok";
  return <span className={`drift-cell ${tone}`}>{value.toFixed(2)}</span>;
}

function FeatureTable({ rows }) {
  // default order is the backend's hybrid-descending ranking
  const [sort, setSort] = useState({ key: "hybrid_drift", dir: -1 });

  const sorted = useMemo(() => {
    const value = (row) =>
      sort.key === "severity" ? SEVERITY_RANK[row.severity] : row[sort.key];
    return [...rows].sort((a, b) => {
      const [va, vb] = [value(a), value(b)];
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  }, [rows, sort]);

  function toggleSort(key) {
    setSort((prev) =>
      prev.key === key ? { key, dir: -prev.dir } : { key, dir: key === "feature" ? 1 : -1 }
    );
  }

  if (rows.length === 0) {
    return <p className="empty">start a simulation to compute per-feature drift</p>;
  }
  return (
    <table className="feature-table">
      <thead>
        <tr>
          {COLUMNS.map((column) => (
            <th key={column.key} onClick={() => toggleSort(column.key)}>
              {column.label}
              {sort.key === column.key && (
                <span className="sort-arrow">{sort.dir === -1 ? "▼" : "▲"}</span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row) => (
          <tr key={row.feature}>
            <td className="feature-name">{row.feature}</td>
            <td><DriftCell value={row.classical_drift} /></td>
            <td><DriftCell value={row.quantum_drift} /></td>
            <td><DriftCell value={row.hybrid_drift} /></td>
            <td><span className={`status-dot ${row.severity}`} title={row.severity} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const ORIGIN_DEST_HINT =
  "Origin = the sender's account, Destination = the recipient's. The distinction " +
  "matters: PaySim fraud typically drains the origin account's balance to zero.";

export default function DriftBreakdown({ latest, featureDrift, driftView }) {
  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">
          {driftView ? "Feature Drift Analysis" : "Drift Breakdown"}
        </span>
        {driftView ? (
          <span className="hint" title={ORIGIN_DEST_HINT}>
            ⓘ origin = sender · destination = recipient
          </span>
        ) : (
          <span className="kpi-icon">▦</span>
        )}
      </div>
      <div className={`breakdown-list${driftView ? "" : " breakdown-aggregate"}`}>
        {driftView ? (
          <FeatureTable rows={featureDrift} />
        ) : (
          <>
            <Row name="Classical — population" sub="PSI · KL · KS · Hellinger, all rows"
                 value={latest ? latest.cds_population : null} />
            <Row name="Classical — fraud subset" sub="same statistics, fraud-labeled rows only"
                 value={latest ? latest.cds_fraud : null} />
            <Row name="Quantum — population" sub="angle-map fidelity kernel, all rows"
                 value={latest ? latest.qds_population : null} />
            <Row name="Quantum — fraud subset" sub="fidelity kernel, fraud-labeled rows only"
                 value={latest ? latest.qds_fraud : null} />
          </>
        )}
      </div>
    </div>
  );
}
