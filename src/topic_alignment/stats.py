"""Paired comparisons between training configurations.

Every comparison is paired twice: the two configurations are evaluated on the same
test items, and they are matched on training seed (which, in the revision protocol,
also fixes the training subset). Uncertainty comes from a two-level bootstrap that
resamples seeds and test items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import canonical_label_ids
from .utils import ensure_dir, resolve_path

RUN_KEYS = ["method", "preference_strategy", "learning_rate"]
CELL_KEYS = ["train_size", "data_allocation"]


def collect_runs(config: dict[str, Any]) -> pd.DataFrame:
    """One row per finished run, read from each run's metadata.json."""
    runs_root = resolve_path(config["project"]["outputs_root"]) / "runs"
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(runs_root.glob("seed_*/train_*/allocation_*/*/metadata.json")):
        run_dir = metadata_path.parent
        predictions_path = run_dir / "predictions_test.csv"
        if not predictions_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        learning_rate = metadata.get("learning_rate_override")
        rows.append(
            {
                "method": str(metadata["method"]),
                "preference_strategy": str(metadata.get("preference_strategy", "na")),
                "learning_rate": "default" if learning_rate is None else f"{float(learning_rate):g}",
                "seed": int(metadata["seed"]),
                "train_size": str(metadata["train_size"]),
                "data_allocation": str(metadata.get("data_allocation", "shared")),
                "predictions_path": str(predictions_path),
            }
        )
    return pd.DataFrame(rows)


def _encode(values: pd.Series, labels: list[str]) -> np.ndarray:
    """Label ids as integers; anything outside the label set becomes len(labels)."""
    mapping = {label: index for index, label in enumerate(labels)}
    return values.map(mapping).fillna(len(labels)).astype(int).to_numpy()


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true == y_pred).mean(axis=-1)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    """Macro F1 over the task classes, vectorised over leading (bootstrap) axes."""
    scores = []
    for class_id in range(class_count):
        true_positive = ((y_true == class_id) & (y_pred == class_id)).sum(axis=-1)
        predicted = (y_pred == class_id).sum(axis=-1)
        actual = (y_true == class_id).sum(axis=-1)
        denominator = predicted + actual
        scores.append(np.where(denominator > 0, 2 * true_positive / np.maximum(denominator, 1), 0.0))
    return np.mean(scores, axis=0)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on two sets of predictions for the same items."""
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    p_value = 1.0 if discordant == 0 else float(binomtest(only_a, discordant, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p_value}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def paired_bootstrap(
    seed_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    class_count: int,
    n_boot: int = 2000,
    random_state: int = 0,
) -> dict[str, dict[str, float]]:
    """Treatment-minus-reference difference in accuracy and macro F1.

    `seed_pairs` holds one (y_true, y_treatment, y_reference) triple per training seed,
    aligned on test items. Each bootstrap replicate resamples seeds, then test items
    within each drawn seed, and averages the difference over the drawn seeds.
    """
    rng = np.random.default_rng(random_state)
    metrics = {
        "accuracy": _accuracy,
        "macro_f1": lambda y_true, y_pred: _macro_f1(y_true, y_pred, class_count),
    }

    def differences(triples, item_index=None):
        output = {name: [] for name in metrics}
        for position, (y_true, y_treatment, y_reference) in enumerate(triples):
            if item_index is not None:
                rows = item_index[position]
                y_true, y_treatment, y_reference = y_true[rows], y_treatment[rows], y_reference[rows]
            for name, metric in metrics.items():
                output[name].append(metric(y_true, y_treatment) - metric(y_true, y_reference))
        return {name: np.mean(values, axis=0) for name, values in output.items()}

    point = differences(seed_pairs)
    seed_count = len(seed_pairs)
    replicates = {name: np.empty(n_boot) for name in metrics}
    for replicate in range(n_boot):
        drawn = rng.integers(0, seed_count, size=seed_count)
        triples = [seed_pairs[index] for index in drawn]
        item_index = [rng.integers(0, len(triple[0]), size=len(triple[0])) for triple in triples]
        for name, value in differences(triples, item_index).items():
            replicates[name][replicate] = value

    result: dict[str, dict[str, float]] = {}
    for name in metrics:
        values = replicates[name]
        p_value = 2 * min((values <= 0).mean(), (values >= 0).mean())
        result[name] = {
            "difference": float(point[name]),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "p_value": float(min(1.0, max(p_value, 1.0 / n_boot))),
        }
    return result


def _load_predictions(path: str, labels: list[str]) -> pd.DataFrame:
    dataframe = pd.read_csv(path, usecols=["example_id", "label", "predicted_label"])
    return pd.DataFrame(
        {
            "example_id": dataframe["example_id"],
            "y_true": _encode(dataframe["label"], labels),
            "y_pred": _encode(dataframe["predicted_label"], labels),
        }
    ).set_index("example_id")


def summary_table(config: dict[str, Any], runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across seeds for every configuration."""
    labels = canonical_label_ids(config)
    rows = []
    for _, run in runs.iterrows():
        predictions = _load_predictions(run["predictions_path"], labels)
        y_true, y_pred = predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()
        rows.append(
            {
                **{key: run[key] for key in [*RUN_KEYS, *CELL_KEYS, "seed"]},
                "accuracy": float(_accuracy(y_true, y_pred)),
                "macro_f1": float(_macro_f1(y_true, y_pred, len(labels))),
            }
        )
    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby([*RUN_KEYS, *CELL_KEYS])[["accuracy", "macro_f1"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    return summary.drop(columns=["macro_f1_count"]).rename(columns={"accuracy_count": "seed_count"})


def contrast_table(
    config: dict[str, Any],
    runs: pd.DataFrame,
    reference_method: str,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Compare every configuration with a reference method, seed by seed.

    With `sft_sft` (continued SFT) as reference the comparison is budget-matched: the
    reference is the run with the same seed, train size and learning rate. With `sft`
    as reference the comparison is against the plain SFT stage.
    """
    labels = canonical_label_ids(config)
    reference_runs = runs[runs["method"] == reference_method]
    match_on_learning_rate = reference_method != "sft"
    rows: list[dict[str, Any]] = []

    treatments = runs[runs["method"] != reference_method]
    for keys, group in treatments.groupby([*RUN_KEYS, *CELL_KEYS]):
        treatment = dict(zip([*RUN_KEYS, *CELL_KEYS], keys))
        if treatment["method"] == "sft":
            continue
        candidates = reference_runs[
            (reference_runs["train_size"] == treatment["train_size"])
            & (reference_runs["data_allocation"] == treatment["data_allocation"])
        ]
        if match_on_learning_rate:
            candidates = candidates[candidates["learning_rate"] == treatment["learning_rate"]]
        reference_by_seed = candidates.set_index("seed")["predictions_path"].to_dict()

        seed_pairs, fixed, broken = [], [], []
        for _, run in group.iterrows():
            if run["seed"] not in reference_by_seed:
                continue
            treatment_df = _load_predictions(run["predictions_path"], labels)
            reference_df = _load_predictions(reference_by_seed[run["seed"]], labels).reindex(treatment_df.index)
            if reference_df["y_pred"].isna().any():
                raise ValueError(f"Test items differ between {run['predictions_path']} and its reference.")
            y_true = treatment_df["y_true"].to_numpy()
            y_treatment = treatment_df["y_pred"].to_numpy()
            y_reference = reference_df["y_pred"].to_numpy().astype(int)
            seed_pairs.append((y_true, y_treatment, y_reference))
            fixed.append(int(((y_treatment == y_true) & (y_reference != y_true)).sum()))
            broken.append(int(((y_treatment != y_true) & (y_reference == y_true)).sum()))
        if not seed_pairs:
            continue

        bootstrap = paired_bootstrap(seed_pairs, len(labels), n_boot=n_boot)
        row: dict[str, Any] = {
            **treatment,
            "reference": reference_method,
            "seed_count": len(seed_pairs),
            "test_items": len(seed_pairs[0][0]),
            # Test items the treatment gets right and the reference wrong, and the reverse.
            "items_fixed_mean": float(np.mean(fixed)),
            "items_broken_mean": float(np.mean(broken)),
        }
        for metric, values in bootstrap.items():
            for name, value in values.items():
                row[f"{metric}_{name}"] = value
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for metric in ["accuracy", "macro_f1"]:
        table[f"{metric}_p_holm"] = holm_adjust(table[f"{metric}_p_value"].tolist())
    return table.sort_values([*CELL_KEYS, "learning_rate", "method", "preference_strategy"]).reset_index(drop=True)


def write_statistics(config: dict[str, Any], n_boot: int = 2000) -> dict[str, str]:
    tables_dir = ensure_dir(resolve_path(config["project"]["results_root"]) / "tables")
    runs = collect_runs(config)
    if runs.empty:
        raise RuntimeError("No finished runs found under the configured outputs_root.")
    outputs: dict[str, str] = {}

    summary_path = tables_dir / "summary_by_configuration.csv"
    summary_table(config, runs).to_csv(summary_path, index=False)
    outputs["summary"] = str(summary_path)

    for reference_method in ["sft_sft", "sft"]:
        if not (runs["method"] == reference_method).any():
            continue
        path = Path(tables_dir) / f"paired_contrasts_vs_{reference_method}.csv"
        contrast_table(config, runs, reference_method, n_boot=n_boot).to_csv(path, index=False)
        outputs[f"contrasts_vs_{reference_method}"] = str(path)
    return outputs
