import numpy as np
import pytest

from quantumguard.config import load_config
from quantumguard.models.train_baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    downsample_negatives,
    evaluate,
    load_paysim,
)

SMOKE_NROWS = 5000


@pytest.fixture(scope="module")
def paysim_sample():
    cfg = load_config()
    csv_path = cfg.path("data_raw_dir") / cfg.baseline_model.source_csv
    if not csv_path.exists():
        pytest.skip(f"PaySim CSV not found at {csv_path}")
    return load_paysim(csv_path, nrows=SMOKE_NROWS)


def test_load_data_has_expected_schema(paysim_sample):
    assert set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["isFraud"]) == set(paysim_sample.columns)
    assert len(paysim_sample) == SMOKE_NROWS
    assert paysim_sample["isFraud"].sum() > 0, "sample must contain at least one fraud row"


def test_train_and_score_end_to_end(paysim_sample):
    cfg = load_config()
    X = paysim_sample[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = paysim_sample["isFraud"]

    X_train, y_train = downsample_negatives(
        X, y, ratio=cfg.baseline_model.train_negative_sample_ratio, random_state=0
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X, y)
    assert 0.0 <= metrics["fraud_precision"] <= 1.0
    assert 0.0 <= metrics["fraud_recall"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
