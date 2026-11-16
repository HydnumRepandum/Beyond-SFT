"""Construction of DPO rejected labels and out-of-fold splits.

Kept free of model code so the logic can be unit-tested without a GPU.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _top_wrong_label(record: dict[str, Any], other_labels: list[str]) -> str:
    return max(other_labels, key=lambda candidate: float(record[f"score__{candidate}"]))


def select_rejected_labels(
    strategy: str,
    gold_label: str,
    labels: list[str],
    rng: random.Random,
    in_sample_record: dict[str, Any] | None = None,
    oof_record: dict[str, Any] | None = None,
) -> list[str]:
    """Return the rejected label(s) paired with the gold label for one example.

    - random_wrong: one uniformly sampled incorrect label.
    - hard_negative: the incorrect label the SFT model trained *on this example*
      scores highest (the strategy of the submitted manuscript).
    - hard_negative_oof: the same, but scored by an SFT model that never saw the
      example, so it reflects a genuine confusion rather than a memorised fit.
    - all_wrong: every incorrect label, giving K-1 pairs per example.
    """
    other_labels = [label for label in labels if label != gold_label]
    if strategy == "random_wrong":
        return [rng.choice(other_labels)]
    if strategy == "all_wrong":
        return other_labels
    if strategy == "hard_negative":
        if in_sample_record is None:
            raise ValueError("hard_negative needs the in-sample SFT scores.")
        return [_top_wrong_label(in_sample_record, other_labels)]
    if strategy == "hard_negative_oof":
        if oof_record is None:
            raise ValueError(
                "hard_negative_oof needs out-of-fold SFT scores. Run `python scripts/train.py oof-score` first."
            )
        return [_top_wrong_label(oof_record, other_labels)]
    raise ValueError(f"Unsupported strategy: {strategy}")


def oof_fold_assignments(dataframe: pd.DataFrame, folds: int, seed: int) -> pd.Series:
    """Stratified fold id per row. The fold count shrinks to the rarest class size."""
    min_class_count = int(dataframe["label"].value_counts().min())
    n_splits = min(int(folds), min_class_count)
    if n_splits < 2:
        raise ValueError(
            "Out-of-fold scoring needs at least 2 examples of every class; "
            f"the rarest class has {min_class_count}."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    assignments = pd.Series(-1, index=dataframe.index, dtype=int)
    for fold_id, (_, held_out) in enumerate(splitter.split(dataframe, dataframe["label"])):
        assignments.iloc[held_out] = fold_id
    return assignments


def rejected_label_agreement(tables: dict[str, pd.DataFrame], reference: str) -> dict[str, float]:
    """Share of examples whose rejected label equals the reference strategy's.

    Reported because single-negative strategies coincide by chance in 1/(K-1) of the
    cases, which bounds how different two strategies can be.
    """
    if reference not in tables:
        return {}
    reference_map = tables[reference].set_index("example_id")["rejected_label"]
    agreement: dict[str, float] = {}
    for strategy, table in tables.items():
        if strategy == reference or table["example_id"].duplicated().any():
            continue
        other = table.set_index("example_id")["rejected_label"]
        shared = reference_map.index.intersection(other.index)
        if len(shared):
            agreement[strategy] = float((reference_map.loc[shared] == other.loc[shared]).mean())
    return agreement
