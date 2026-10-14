# QuantumGuard

Drift monitoring and semi-automated retraining for a production fraud-detection
model. A FastAPI service scores transactions and continuously computes a
**Model Health Score (MHS)** fusing classical drift statistics (PSI, KL, KS,
Hellinger — computed on both the full population and the fraud-labeled subset)
with a **quantum-inspired kernel drift signal** (PennyLane feature-map kernel on
PCA-reduced, subsampled windows). A deterministic **Trend Detector** (rolling
slope + CUSUM) decides when to retrain based on the MHS trajectory; a
**Narration Layer** describes each decision in plain language. Triggered
retraining produces a versioned candidate model, evaluates it offline,
validates it in **shadow mode** against replayed traffic, and requires
administrator **approval on a React dashboard** before deployment.

## Headline result

The benchmark asks one question: *does the quantum-inspired kernel add
anything over classical baselines?* The answer, with numbers
([benchmark/REPORT.md](benchmark/REPORT.md) / `REPORT.pdf`):

- **Population-level drift: no.** Classical per-feature statistics match or
  beat every kernel method, and on fraud-subset-only drift viewed at population
  level they are dramatically better (delay 1 vs 16–29 windows).
- **Small-sample fraud-subset monitoring: yes.** On gradual drift over ~40-row
  fraud-subset windows, the quantum kernels detect at **delay 1** vs 10
  (MMD-RBF, domain classifier) and 19 (classical CDS).
- **Entanglement does not help.** The angle embedding — the no-entanglement
  control, classically simulable — ties ZZ/IQP on delay, edges it on
  separation, and runs 6.5× faster. No quantum advantage is claimed;
  "quantum-inspired" means the quantum formalism guided a good kernel choice.
- **Cost:** the selected angle map is the cheapest multivariate detector in
  the suite (~13–20 ms per 500-row window).

## Repository layout

```
configs/default.yaml         # every weight, threshold, window size, qubit cap
src/quantumguard/
  drift/                     # classical.py, quantum_kernel.py, baselines.py,
                             # preprocess.py, injection.py, base.py (protocol)
  health/                    # mhs.py (fusion), trend.py (pure trigger logic)
  narration/                 # templated narration; LLM rendering behind a flag
  retrain/                   # pipeline.py (blend/train/version), shadow.py
  api/                       # FastAPI app, WebSocket manager, simulation engine
  models/                    # training, registry (versioned .pkl store)
  storage/db.py              # single SQLite access layer
  benchmark/metrics.py       # detection delay, FAR, calibration, split assert
benchmark/                   # scenario generation, benchmark, ablation, reports
dashboard/                   # React + Vite + Recharts admin dashboard
tests/                       # 90+ tests incl. golden trend-series fixtures
```

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) (Python 3.11 is pinned and
installed automatically), Node 18+ (dashboard only).

```sh
uv sync --all-extras          # python deps (--extra llm for LLM narration)
```

**Data:** download the PaySim dataset (`PS_20174392719_1491204439457_log.csv`,
[Kaggle: ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)) and
place it in `data/raw/`. Then train the baseline model:

```sh
uv run python -m quantumguard.models.train_baseline --version v1
uv run pytest                 # everything should pass
```

## Reproducing the benchmark

The held-out / evaluation scenario split lives in `configs/default.yaml`
(`benchmark:` section) and is **asserted in code** — the run aborts if the sets
overlap. Thresholds are calibrated only on held-out pre-drift windows; the
quantum feature map is selected on held-out scenarios only.

```sh
# 1. Generate the six drift scenarios (sudden x2, gradual x2, fraud-subset, control)
uv run python benchmark/inject_drift.py

# 2. Score every detector on every scenario, calibrate, evaluate, write report
#    (~5 min; cached in score_series.csv, add --recompute to refresh)
uv run python benchmark/run_benchmark.py --recompute

# 3. Feature-map ablation on the held-out slice (angle / ZZ / re-uploading)
uv run python benchmark/ablation_feature_maps.py

# 4. Optional artifacts
uv run python benchmark/export_pdf.py           # benchmark/REPORT.pdf
uv run python benchmark/plot_cds_demo.py        # phase-1 CDS demo plots
uv run python benchmark/run_health_demo.py      # MHS + trigger-timing plots
uv run python benchmark/run_retrain_loop.py     # full headless retrain loop
uv run python benchmark/profile_detectors.py    # wall-clock per window
```

Outputs land in `benchmark/`: `results.csv`, `ablation_results.csv`,
`thresholds.json`, `REPORT.md` (embeds `INTERPRETATION.md`), `REPORT.pdf`,
and plots under `benchmark/plots/`.

## Live dashboard

```sh
cd dashboard && npm install && npm run build && cd ..
uv run uvicorn quantumguard.api.main:app --port 8000
```

Open http://localhost:8000, pick a scenario, and press **Start simulation**.
The stream plays a drift scenario window-by-window: live predictions, MHS with
status bands, population-vs-fraud-subset drift charts, and a narrated
reasoning log. When the trend detector triggers, a candidate is retrained and
shadow-validated, then appears as a **pending** card with offline
candidate-vs-deployed metrics and shadow agreement — approve to deploy (the
engine hot-reloads the model mid-stream and re-anchors health baselines) or
reject. Nothing ever auto-deploys.

For dashboard development: `npm run dev` (proxies `/api` and `/ws` to :8000).

## Design invariants (enforced, not aspirational)

- Every detector implements the `DriftDetector` protocol:
  `score(reference, current) -> float in [0, 1]`.
- All weights, thresholds, window sizes and qubit/depth caps live in
  `configs/default.yaml` — never in module code.
- Quantum kernel caps (16 qubits, 2 layers, ≤300 samples/window) are
  methodological (kernel concentration), enforced at runtime.
- Drift statistics are always computed twice: population AND fraud subset.
- Ground-truth labels arrive with a simulated lag (`label_latency_windows`).
- The trend detector is a pure function with golden regression tests; no LLM
  call exists anywhere in the decision path — narration is downstream-only.
- Models are versioned (`Model_v{N}.pkl` + `registry.json`); retraining
  creates a *pending* candidate and never deploys.
- SQLite goes through `storage/db.py` only.

## LLM narration (optional)

`narration.llm_enabled: false` by default; the deterministic template is
canonical. To enable the LLM rendering (Claude, via the `anthropic` SDK):
`uv sync --extra llm`, set Anthropic API credentials in the environment, flip
the config flag, and call `narrate_llm(decision, components)`. The LLM
receives the frozen decision record only and cannot influence any decision.

## Extending

- **IEEE-CIS as a second dataset:** the detector stack is dataset-agnostic —
  everything downstream of `drift/injection.py` consumes numeric feature
  matrices. Port by adding a loader with IEEE-CIS transaction features, a
  `baseline_model` config entry, and scenario definitions; the benchmark
  machinery runs unchanged. (Requires accepting the Kaggle competition terms
  to download.)
- **New detectors:** implement the protocol in `drift/base.py`, add the
  synthetic-drift unit test battery from `tests/test_detectors.py`, and the
  benchmark picks it up via `build_detectors()` in `benchmark/run_benchmark.py`.
