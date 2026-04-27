from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .config import canonical_label_ids


def compute_metrics(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if dataframe.empty:
        return {
            "row_count": 0,
            "accuracy": None,
            "macro_f1": None,
            "macro_auc": None,
            "per_class_f1": {},
            "per_class_auc": {},
            "confusion_matrix": [],
            "invalid_prediction_count": 0,
        }

    labels = canonical_label_ids(config)
    y_true = dataframe["label"].tolist()
    invalid_mask = ~dataframe["predicted_label"].isin(labels)
    y_pred = dataframe["predicted_label"].where(~invalid_mask, "__INVALID__").tolist()

    per_class_scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_auc: dict[str, float | None] = {}
    valid_auc_scores: list[float] = []
    for label in labels:
        score_column = f"score__{label}"
        if score_column not in dataframe.columns:
            per_class_auc[label] = None
            continue

        binary_true = dataframe["label"].eq(label).astype(int)
        if binary_true.nunique(dropna=False) < 2:
            per_class_auc[label] = None
            continue

        auc_score = float(roc_auc_score(binary_true, dataframe[score_column].astype(float)))
        per_class_auc[label] = auc_score
        valid_auc_scores.append(auc_score)

    return {
        "row_count": int(len(dataframe)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_auc": float(sum(valid_auc_scores) / len(valid_auc_scores)) if valid_auc_scores else None,
        "per_class_f1": {label: float(score) for label, score in zip(labels, per_class_scores.tolist())},
        "per_class_auc": per_class_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "invalid_prediction_count": int(invalid_mask.sum()),
    }


def compute_metrics_bundle(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    payload = {"all_examples": compute_metrics(dataframe, config)}
    if "is_hard_primary" in dataframe.columns:
        payload["hard_primary"] = compute_metrics(dataframe[dataframe["is_hard_primary"]].copy(), config)
    if "is_hard_secondary" in dataframe.columns:
        payload["hard_secondary"] = compute_metrics(dataframe[dataframe["is_hard_secondary"]].copy(), config)
    return payload
