import React from "react";

export default function PredictionsPanel({ predictions, highlight }) {
  return (
    <div className={`card ${highlight ? "highlight" : ""}`}>
      <div className="card-head">
        <span className="card-title">Live Predictions</span>
        <span className="kpi-icon">⚡</span>
      </div>
      <div className="scroll-body">
        {predictions.length === 0 ? (
          <p className="empty">start a simulation to stream predictions</p>
        ) : (
          <table className="preds">
            <thead>
              <tr>
                <th>win</th>
                <th>type</th>
                <th>amount</th>
                <th>fraud prob.</th>
                <th>label</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((row) => (
                <tr key={row.key}>
                  <td>{row.window_id}</td>
                  <td>{row.type}</td>
                  <td>{row.amount.toLocaleString()}</td>
                  <td>
                    <span className={`proba-pill ${row.proba >= 0.5 ? "hot" : ""}`}>
                      {row.proba.toFixed(3)}
                    </span>
                  </td>
                  <td>{row.is_fraud ? "fraud" : "legit"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
