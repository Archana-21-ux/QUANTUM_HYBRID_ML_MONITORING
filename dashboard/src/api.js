async function json(response) {
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

export const api = {
  simulationStatus: () => fetch("/api/simulation/status").then(json),
  startSimulation: (scenario) =>
    fetch("/api/simulation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario }),
    }).then(json),
  stopSimulation: () => fetch("/api/simulation/stop", { method: "POST" }).then(json),
  candidates: () => fetch("/api/retrain/candidates").then(json),
  uploadScenario: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/scenarios/upload", { method: "POST", body: form }).then(json);
  },
  approve: (id) =>
    fetch(`/api/retrain/${id}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decided_by: "admin" }),
    }).then(json),
  reject: (id) =>
    fetch(`/api/retrain/${id}/reject`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decided_by: "admin" }),
    }).then(json),
};
