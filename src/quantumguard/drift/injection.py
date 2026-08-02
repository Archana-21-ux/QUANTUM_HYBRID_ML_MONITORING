from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantumguard.config import Config, load_config

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
STREAM_COLUMNS = ["window_id", "type"] + NUMERIC_FEATURES + ["isFraud"]

DRIFT_KINDS = ("sudden", "gradual", "fraud_subset", "none")


def build_stream(
    pool: pd.DataFrame,
    *,
    n_rows: int,
    fraud_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Draw a simulated transaction stream from a data pool with a boosted
    fraud rate (raw PaySim fraud is ~0.13%, far too sparse for per-window
    fraud-subset statistics)."""
    rng = np.random.default_rng(seed)
    fraud_pool = pool[pool["isFraud"] == 1]
    legit_pool = pool[pool["isFraud"] == 0]

    n_fraud = int(round(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud
    if n_fraud > len(fraud_pool):
        raise ValueError(f"need {n_fraud} fraud rows but pool only has {len(fraud_pool)}")

    fraud_rows = fraud_pool.sample(n=n_fraud, random_state=rng.integers(2**31))
    legit_rows = legit_pool.sample(n=n_legit, random_state=rng.integers(2**31))
    stream = pd.concat([fraud_rows, legit_rows])
    return stream.sample(frac=1.0, random_state=rng.integers(2**31)).reset_index(drop=True)


def drift_intensity(
    window_id: np.ndarray,
    *,
    kind: str,
    drift_start_window: int,
    ramp_windows: int,
) -> np.ndarray:
    """Per-row drift intensity t in [0, 1] as a function of window index.

    sudden / fraud_subset: step from 0 to 1 at the drift start.
    gradual: linear ramp from 0 to 1 over `ramp_windows` after the start
        (apply_drift converts this to a per-row regime-mixing probability).
    none: always 0.
    """
    if kind not in DRIFT_KINDS:
        raise ValueError(f"unknown drift kind {kind!r}, expected one of {DRIFT_KINDS}")
    if kind == "none":
        return np.zeros(len(window_id), dtype=float)
    past_start = window_id >= drift_start_window
    if kind in ("sudden", "fraud_subset"):
        return past_start.astype(float)
    return np.clip((window_id - drift_start_window + 1) / ramp_windows, 0.0, 1.0) * past_start


def apply_drift(
    stream: pd.DataFrame,
    *,
    kind: str,
    drift_start_window: int,
    ramp_windows: int,
    scale: float,
    offset_std: float,
    shift_features: list[str],
    feature_stds: dict[str, float],
    seed: int = 0,
) -> pd.DataFrame:
    """Apply the configured covariate shift to a stream (population-wide, or
    fraud rows only for kind='fraud_subset'). x' = x*(1+(scale-1)*t) + offset_std*std*t.

    Gradual drift is regime mixing: each row takes the FULL shift with
    probability equal to the ramp intensity of its window. A uniform partial
    shift instead would not ramp — on heavy-tailed features like PaySim
    amounts, even a few percent of a std applied to every row saturates the
    drift statistics immediately.
    """
    stream = stream.copy()
    t = drift_intensity(
        stream["window_id"].to_numpy(),
        kind=kind,
        drift_start_window=drift_start_window,
        ramp_windows=ramp_windows,
    )
    if kind == "gradual":
        rng = np.random.default_rng(seed)
        t = (rng.random(len(t)) < t).astype(float)
    if kind == "fraud_subset":
        t = t * (stream["isFraud"].to_numpy() == 1)

    for feature in shift_features:
        x = stream[feature].to_numpy(dtype=float)
        stream[feature] = x * (1.0 + (scale - 1.0) * t) + offset_std * feature_stds[feature] * t
    return stream


def generate_scenario(
    pool: pd.DataFrame,
    name: str,
    params: dict,
    *,
    config: Config | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Generate one scenario: an undrifted reference sample plus a drifted
    stream, written to <output_dir>/<name>/ as reference.csv, stream.csv, meta.json."""
    cfg = config if config is not None else load_config()
    di = cfg.drift_injection
    window_size = cfg.windowing.window_size
    out = (output_dir if output_dir is not None else cfg.path("drift_scenarios_dir")) / name
    out.mkdir(parents=True, exist_ok=True)

    seed = params["seed"]
    reference = build_stream(
        pool,
        n_rows=di.reference_windows * window_size,
        fraud_rate=di.stream_fraud_rate,
        seed=seed,
    )
    reference.insert(0, "window_id", reference.index // window_size)

    stream = build_stream(
        pool,
        n_rows=di.n_windows * window_size,
        fraud_rate=di.stream_fraud_rate,
        seed=seed + 1,
    )
    stream.insert(0, "window_id", stream.index // window_size)

    feature_stds = {f: float(reference[f].std()) for f in di.shift_features}
    stream = apply_drift(
        stream,
        kind=params["kind"],
        drift_start_window=di.drift_start_window,
        ramp_windows=di.gradual_ramp_windows,
        scale=params["scale"],
        offset_std=params["offset_std"],
        shift_features=list(di.shift_features),
        feature_stds=feature_stds,
        seed=seed + 2,
    )

    reference[STREAM_COLUMNS].to_csv(out / "reference.csv", index=False)
    stream[STREAM_COLUMNS].to_csv(out / "stream.csv", index=False)
    meta = {
        "name": name,
        "kind": params["kind"],
        "seed": seed,
        "scale": params["scale"],
        "offset_std": params["offset_std"],
        "drift_start_window": di.drift_start_window,
        "gradual_ramp_windows": di.gradual_ramp_windows,
        "n_windows": di.n_windows,
        "window_size": window_size,
        "stream_fraud_rate": di.stream_fraud_rate,
        "shift_features": list(di.shift_features),
        "feature_stds": feature_stds,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out


def load_scenario(name: str, config: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load (reference, stream, meta) for a generated scenario."""
    cfg = config if config is not None else load_config()
    scenario_dir = cfg.path("drift_scenarios_dir") / name
    reference = pd.read_csv(scenario_dir / "reference.csv")
    stream = pd.read_csv(scenario_dir / "stream.csv")
    with open(scenario_dir / "meta.json") as f:
        meta = json.load(f)
    return reference, stream, meta


def generate_all(config: Config | None = None) -> list[Path]:
    """Generate every scenario declared in configs drift_injection.scenarios."""
    cfg = config if config is not None else load_config()
    csv_path = cfg.path("data_raw_dir") / cfg.baseline_model.source_csv
    pool = pd.read_csv(csv_path, usecols=["type"] + NUMERIC_FEATURES + ["isFraud"])
    return [
        generate_scenario(pool, name, params, config=cfg)
        for name, params in cfg.drift_injection.scenarios.items()
    ]
