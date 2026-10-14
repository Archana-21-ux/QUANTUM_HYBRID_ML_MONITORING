from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from quantumguard.config import load_config
from quantumguard.models.registry import register_model

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def load_paysim(csv_path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        csv_path, usecols=CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["isFraud"], nrows=nrows
    )


def build_pipeline(cfg) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=cfg.baseline_model.n_estimators,
        max_depth=cfg.baseline_model.max_depth,
        learning_rate=cfg.baseline_model.learning_rate,
        random_state=cfg.baseline_model.random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("classify", classifier)])


def downsample_negatives(X_train: pd.DataFrame, y_train: pd.Series, ratio: float, random_state: int):
    """Keep all positives; subsample negatives to `ratio` legit-per-fraud. Training-only."""
    pos_idx = y_train[y_train == 1].index
    neg_idx = y_train[y_train == 0].index
    n_neg_keep = min(len(neg_idx), int(len(pos_idx) * ratio))
    neg_idx_sampled = pd.Index(neg_idx).to_series().sample(n=n_neg_keep, random_state=random_state).index
    keep_idx = pos_idx.union(neg_idx_sampled)
    return X_train.loc[keep_idx], y_train.loc[keep_idx]


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "fraud_precision": float(precision),
        "fraud_recall": float(recall),
        "fraud_f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "n_test": int(len(y_test)),
        "n_test_fraud": int(y_test.sum()),
    }


def train(version: str = "v1", nrows: int | None = None) -> dict:
    cfg = load_config()
    csv_path = cfg.path("data_raw_dir") / cfg.baseline_model.source_csv

    df = load_paysim(csv_path, nrows=nrows)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.baseline_model.test_size,
        random_state=cfg.baseline_model.random_state,
        stratify=y,
    )
    X_train, y_train = downsample_negatives(
        X_train,
        y_train,
        ratio=cfg.baseline_model.train_negative_sample_ratio,
        random_state=cfg.baseline_model.random_state,
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)

    model_dir = cfg.path("model_dir")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"Model_{version}.pkl"
    joblib.dump(pipeline, model_dir / model_filename)

    register_model(
        version=version,
        model_filename=model_filename,
        metrics=metrics,
        trained_on=cfg.baseline_model.source_csv,
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    result_metrics = train(version=args.version)
    for key, value in result_metrics.items():
        print(f"{key}: {value}")
