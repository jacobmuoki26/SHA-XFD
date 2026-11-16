from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from claim_data import build_features, dataset_sha256, validate_claims_dataset

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "SHA_Claims_10000_Dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "sha_xfd_model.pkl"


def train(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    source_hash = dataset_sha256(dataset_path)
    data = pd.read_csv(dataset_path, keep_default_na=False)
    validate_claims_dataset(data)
    labels = data["IsFraud"].astype("int8")
    class_counts = labels.value_counts()
    if class_counts.min() < 5:
        raise ValueError("Training requires at least five claims in each target class.")

    features = build_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=31,
        learning_rate=0.1,
        random_state=42,
        class_weight="balanced",
        early_stopping=False,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "evaluation_note": (
            "Holdout metrics on synthetic development data; not evidence of "
            "real-world fraud-detection performance."
        ),
    }
    bundle = {
        "model": model,
        "features": list(features.columns),
        "diagnoses": sorted(data["Diagnosis"].astype(str).unique().tolist()),
        "metrics": metrics,
        "dataset_sha256": source_hash,
    }
    if dataset_sha256(dataset_path) != source_hash:
        raise OSError("Dataset changed during training; no model artifact was saved.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=model_path.parent, delete=False
        ) as model_file:
            temporary_path = model_file.name
            pickle.dump(bundle, model_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ):
        print(f"{metric}: {metrics[metric]:.4f}")
    print(metrics["evaluation_note"])
    print(f"Saved model bundle to {model_path}")
    return metrics


if __name__ == "__main__":
    train()
