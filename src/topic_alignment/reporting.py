from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import canonical_label_ids, data_allocation_variants, reference_train_size
from .evaluation import compute_metrics
from .utils import ensure_dir, resolve_path, write_json


_CURVE_METHOD_ORDER = [
    "Pretrained baseline",
    "SFT",
    "DPO (random wrong)",
    "DPO (hard negative)",
    "KTO",
    "SFT->DPO (random wrong)",
    "SFT->DPO (hard negative)",
    "SFT->KTO",
]

_CURVE_PALETTE = {
    "Pretrained baseline": "#4C78A8",
    "SFT": "#222222",
    "DPO (random wrong)": "#72B7B2",
    "DPO (hard negative)": "#B279A2",
    "KTO": "#9D755D",
    "SFT->DPO (random wrong)": "#F58518",
    "SFT->DPO (hard negative)": "#E45756",
    "SFT->KTO": "#54A24B",
}


def _baseline_predictions_path(config: dict[str, Any]) -> Path:
    return resolve_path(config["project"]["outputs_root"]) / "runs" / "baseline" / "predictions_test.csv"


def _allocation_root(config: dict[str, Any], seed: int, train_size: int | str, data_allocation: str) -> Path:
    return (
        resolve_path(config["project"]["outputs_root"])
        / "runs"
        / f"seed_{seed}"
        / f"train_{str(train_size).lower()}"
        / f"allocation_{str(data_allocation).lower()}"
    )


def _sft_dir(config: dict[str, Any], seed: int, train_size: int | str, data_allocation: str) -> Path:
    return _allocation_root(config, seed, train_size, data_allocation) / "sft"


def _preferences_dir(config: dict[str, Any], seed: int, train_size: int | str, data_allocation: str) -> Path:
    return _allocation_root(config, seed, train_size, data_allocation) / "preferences"


def _dpo_dir(config: dict[str, Any], seed: int, train_size: int | str, data_allocation: str, strategy: str, start_from: str) -> Path:
    return _allocation_root(config, seed, train_size, data_allocation) / f"dpo_{strategy}_from_{start_from}"


def _kto_dir(config: dict[str, Any], seed: int, train_size: int | str, data_allocation: str, strategy: str, start_from: str) -> Path:
    return _allocation_root(config, seed, train_size, data_allocation) / f"kto_{strategy}_from_{start_from}"


def _kto_strategy(config: dict[str, Any]) -> str:
    return str(config.get("kto", {}).get("dataset_strategy", "all_wrong"))


def _display_method_label(method: str, data_allocation: str) -> str:
    if data_allocation in {"na", "shared"}:
        return method
    if data_allocation == "sft25_dpo75":
        return f"{method} (25/75 split)"
    return f"{method} ({data_allocation})"


def _curve_display_label(method: str, preference_strategy: str) -> str | None:
    if method == "pretrained_baseline":
        return "Pretrained baseline"
    if method == "sft":
        return "SFT"
    if method == "dpo":
        if preference_strategy == "random_wrong":
            return "DPO (random wrong)"
        if preference_strategy == "hard_negative":
            return "DPO (hard negative)"
        return None
    if method == "kto":
        return "KTO"
    if method == "sft_dpo":
        if preference_strategy == "random_wrong":
            return "SFT->DPO (random wrong)"
        if preference_strategy == "hard_negative":
            return "SFT->DPO (hard negative)"
        return None
    if method == "sft_kto":
        return "SFT->KTO"
    return None


def _load_with_tags(predictions_path: Path, tags_path: Path | None = None) -> pd.DataFrame:
    predictions_df = pd.read_csv(predictions_path)
    if tags_path is not None and not {"is_hard_primary", "is_hard_secondary"}.issubset(predictions_df.columns):
        tags_df = pd.read_csv(tags_path)[["example_id", "is_hard_primary", "is_hard_secondary"]]
        predictions_df = predictions_df.merge(tags_df, on="example_id", how="left")
    return _normalize_hard_tag_columns(predictions_df)


def _normalize_hard_tag_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    for column in ["is_hard_primary", "is_hard_secondary"]:
        candidate_columns = [name for name in [column, f"{column}_x", f"{column}_y"] if name in normalized.columns]
        if not candidate_columns:
            normalized[column] = False
            continue

        series = normalized[candidate_columns[0]]
        for candidate in candidate_columns[1:]:
            series = series.combine_first(normalized[candidate])
        normalized[column] = series.fillna(False).astype(bool)

        for candidate in candidate_columns:
            if candidate != column:
                normalized = normalized.drop(columns=candidate)
    return normalized


def _subset_frames(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    normalized = _normalize_hard_tag_columns(dataframe)
    return {
        "all_examples": normalized,
    }


def build_results_table(config: dict[str, Any]) -> pd.DataFrame:
    processed_dir = resolve_path(config["project"]["data_root"]) / config["task"]["name"]
    manifest = pd.read_json(processed_dir / "manifest.json", typ="series").to_dict()
    available_train_sizes = manifest["available_train_sizes"]
    allocations = data_allocation_variants(config)
    kto_strategy = _kto_strategy(config)
    rows: list[dict[str, Any]] = []

    baseline_df = pd.read_csv(_baseline_predictions_path(config))
    baseline_all_metrics = compute_metrics(baseline_df, config)
    rows.append(
        {
            "method": "pretrained_baseline",
            "train_size": "na",
            "seed": "na",
            "preference_strategy": "na",
            "data_allocation": "na",
            "subset": "all_examples",
            "accuracy": baseline_all_metrics["accuracy"],
            "macro_f1": baseline_all_metrics["macro_f1"],
            "macro_auc": baseline_all_metrics["macro_auc"],
        }
    )

    for seed in config["task"]["subset_seeds"]:
        for train_size in available_train_sizes:
            for data_allocation in allocations:
                sft_dir = _sft_dir(config, seed, train_size, data_allocation)
                tagged_test_path = sft_dir / "tagged_test_predictions.csv"
                if not tagged_test_path.exists():
                    continue
                sft_test_df = _normalize_hard_tag_columns(pd.read_csv(tagged_test_path))
                for subset_name, subset_df in _subset_frames(sft_test_df).items():
                    metric_bundle = compute_metrics(subset_df, config)
                    rows.append(
                        {
                            "method": "sft",
                            "train_size": str(train_size),
                            "seed": seed,
                            "preference_strategy": "na",
                            "data_allocation": data_allocation,
                            "subset": subset_name,
                            "accuracy": metric_bundle["accuracy"],
                            "macro_f1": metric_bundle["macro_f1"],
                            "macro_auc": metric_bundle["macro_auc"],
                        }
                    )

                baseline_joined = baseline_df.merge(
                    sft_test_df[["example_id", "is_hard_primary", "is_hard_secondary"]],
                    on="example_id",
                    how="left",
                )
                baseline_joined = _normalize_hard_tag_columns(baseline_joined)
                for subset_name, subset_df in _subset_frames(baseline_joined).items():
                    metric_bundle = compute_metrics(subset_df, config)
                    rows.append(
                        {
                            "method": "pretrained_baseline",
                            "train_size": str(train_size),
                            "seed": seed,
                            "preference_strategy": "na",
                            "data_allocation": data_allocation,
                            "subset": subset_name,
                            "accuracy": metric_bundle["accuracy"],
                            "macro_f1": metric_bundle["macro_f1"],
                            "macro_auc": metric_bundle["macro_auc"],
                        }
                    )

                for strategy in config["dpo"]["strategies"]:
                    dpo_run_specs = [("sft", "sft_dpo")]
                    if data_allocation == "shared":
                        dpo_run_specs.insert(0, ("base", "dpo"))
                    for start_from, method_name in dpo_run_specs:
                        predictions_path = _dpo_dir(config, seed, train_size, data_allocation, strategy, start_from) / "predictions_test.csv"
                        if not predictions_path.exists():
                            continue
                        merged_df = _load_with_tags(
                            predictions_path,
                            sft_dir / "tagged_test_predictions.csv",
                        )
                        for subset_name, subset_df in _subset_frames(merged_df).items():
                            metric_bundle = compute_metrics(subset_df, config)
                            rows.append(
                                {
                                    "method": method_name,
                                    "train_size": str(train_size),
                                    "seed": seed,
                                    "preference_strategy": strategy,
                                    "data_allocation": data_allocation,
                                    "subset": subset_name,
                                    "accuracy": metric_bundle["accuracy"],
                                    "macro_f1": metric_bundle["macro_f1"],
                                    "macro_auc": metric_bundle["macro_auc"],
                                }
                            )
                kto_run_specs = [("sft", "sft_kto")]
                if data_allocation == "shared":
                    kto_run_specs.insert(0, ("base", "kto"))
                for start_from, method_name in kto_run_specs:
                    predictions_path = _kto_dir(config, seed, train_size, data_allocation, kto_strategy, start_from) / "predictions_test.csv"
                    if not predictions_path.exists():
                        continue
                    merged_df = _load_with_tags(
                        predictions_path,
                        sft_dir / "tagged_test_predictions.csv",
                    )
                    for subset_name, subset_df in _subset_frames(merged_df).items():
                        metric_bundle = compute_metrics(subset_df, config)
                        rows.append(
                                {
                                    "method": method_name,
                                    "train_size": str(train_size),
                                    "seed": seed,
                                    "preference_strategy": kto_strategy,
                                    "data_allocation": data_allocation,
                                    "subset": subset_name,
                                    "accuracy": metric_bundle["accuracy"],
                                    "macro_f1": metric_bundle["macro_f1"],
                                    "macro_auc": metric_bundle["macro_auc"],
                                }
                            )
    return pd.DataFrame(rows)


def write_results_tables(config: dict[str, Any], results_df: pd.DataFrame) -> dict[str, str]:
    tables_dir = ensure_dir(resolve_path(config["project"]["results_root"]) / "tables")
    raw_path = tables_dir / "aggregated_results.csv"
    results_df.to_csv(raw_path, index=False)

    summary = (
        results_df.groupby(
            ["method", "preference_strategy", "data_allocation", "train_size", "subset"],
            dropna=False,
        )[["accuracy", "macro_f1", "macro_auc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "method",
        "preference_strategy",
        "data_allocation",
        "train_size",
        "subset",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_auc_mean",
        "macro_auc_std",
    ]
    summary_path = tables_dir / "aggregated_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"raw_table": str(raw_path), "summary_table": str(summary_path)}


def plot_training_curves(config: dict[str, Any], results_df: pd.DataFrame) -> list[str]:
    figures_dir = ensure_dir(resolve_path(config["project"]["results_root"]) / "figures")
    for old_path in figures_dir.glob("training_curve_*.png"):
        old_path.unlink()

    plot_df = results_df[
        (results_df["subset"] == "all_examples")
        & (results_df["method"].isin(["pretrained_baseline", "sft", "dpo", "kto", "sft_dpo", "sft_kto"]))
        & (results_df["train_size"] != "na")
    ].copy()
    plot_df = plot_df.drop_duplicates(
        subset=["method", "preference_strategy", "data_allocation", "train_size", "seed", "subset"]
    )
    plot_df["display_method"] = plot_df.apply(
        lambda row: _curve_display_label(str(row["method"]), str(row["preference_strategy"])),
        axis=1,
    )
    plot_df = plot_df.dropna(subset=["display_method"])
    plot_df["train_size_numeric"] = plot_df["train_size"].replace({"full": 10_000}).astype(int)
    plot_df = plot_df.sort_values("train_size_numeric")

    allocation_specs = [
        ("shared", "Shared dataset"),
        ("sft25_dpo75", "25/75 split"),
    ]

    saved_paths: list[str] = []
    sns.set_theme(style="whitegrid")
    for data_allocation, allocation_title in allocation_specs:
        allocation_df = plot_df[plot_df["data_allocation"] == data_allocation].copy()
        if allocation_df.empty:
            continue

        xticks = sorted(allocation_df["train_size_numeric"].unique())
        xtick_labels = ["full" if value == 10_000 else str(int(value)) for value in xticks]
        for metric in ["accuracy", "macro_f1"]:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            sns.lineplot(
                data=allocation_df,
                x="train_size_numeric",
                y=metric,
                hue="display_method",
                hue_order=[label for label in _CURVE_METHOD_ORDER if label in set(allocation_df["display_method"])],
                palette=_CURVE_PALETTE,
                marker="o",
                estimator="mean",
                errorbar="sd",
                linewidth=2.4,
                ax=ax,
            )
            ax.set_xlabel("Train size")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} by train size ({allocation_title})")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                title=None,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=(0, 0, 0.8, 1))
            output_path = figures_dir / f"training_curve_{metric}_{data_allocation}.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(output_path))
    return saved_paths


def _full_train_run_specs(
    config: dict[str, Any],
    seed: int,
    full_train: int | str,
    data_allocation: str,
) -> list[tuple[str, Path]]:
    primary_strategy = config["dpo"]["primary_preference_strategy"]
    kto_strategy = _kto_strategy(config)
    run_specs = [
        ("pretrained_baseline", _baseline_predictions_path(config)),
        ("sft", _sft_dir(config, seed, full_train, data_allocation) / "tagged_test_predictions.csv"),
    ]
    if data_allocation == "shared":
        run_specs.extend(
            [
                ("dpo", _dpo_dir(config, seed, full_train, data_allocation, primary_strategy, "base") / "predictions_test.csv"),
                ("kto", _kto_dir(config, seed, full_train, data_allocation, kto_strategy, "base") / "predictions_test.csv"),
            ]
        )
    run_specs.extend(
        [
            ("sft_dpo", _dpo_dir(config, seed, full_train, data_allocation, primary_strategy, "sft") / "predictions_test.csv"),
            ("sft_kto", _kto_dir(config, seed, full_train, data_allocation, kto_strategy, "sft") / "predictions_test.csv"),
        ]
    )
    return run_specs


def write_per_class_breakdown(config: dict[str, Any]) -> str:
    tables_dir = ensure_dir(resolve_path(config["project"]["results_root"]) / "tables")
    allocations = data_allocation_variants(config)
    available_train_sizes = pd.read_json(
        resolve_path(config["project"]["data_root"]) / config["task"]["name"] / "manifest.json",
        typ="series",
    ).to_dict()["available_train_sizes"]
    full_train = reference_train_size(config, available_train_sizes)

    rows: list[dict[str, Any]] = []
    for seed in config["task"]["subset_seeds"]:
        for data_allocation in allocations:
            for method, path in _full_train_run_specs(config, seed, full_train, data_allocation):
                if not path.exists():
                    continue
                dataframe = pd.read_csv(path)
                metrics = compute_metrics(dataframe, config)
                for label in metrics["per_class_f1"]:
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "data_allocation": data_allocation,
                            "label": label,
                            "f1": metrics["per_class_f1"][label],
                            "auc": metrics["per_class_auc"].get(label),
                        }
                    )

    output_path = tables_dir / "per_class_f1_full_train.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path)


def plot_confusion_matrices(config: dict[str, Any]) -> list[str]:
    figures_dir = ensure_dir(resolve_path(config["project"]["results_root"]) / "figures")
    labels = canonical_label_ids(config)
    allocations = data_allocation_variants(config)
    available_train_sizes = pd.read_json(
        resolve_path(config["project"]["data_root"]) / config["task"]["name"] / "manifest.json",
        typ="series",
    ).to_dict()["available_train_sizes"]
    full_train = reference_train_size(config, available_train_sizes)

    for old_path in figures_dir.glob("confusion_*.png"):
        old_path.unlink()

    figure_paths: list[str] = []
    for seed in config["task"]["subset_seeds"]:
        for data_allocation in allocations:
            for method, path in _full_train_run_specs(config, seed, full_train, data_allocation):
                if not path.exists():
                    continue
                dataframe = pd.read_csv(path)
                metrics = compute_metrics(dataframe, config)
                matrix = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
                plt.figure(figsize=(7, 6))
                sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion matrix: {_display_method_label(method, data_allocation)}, seed {seed}")
                plt.ylabel("Gold label")
                plt.xlabel("Predicted label")
                plt.tight_layout()
                output_path = figures_dir / f"confusion_{method}_{data_allocation}_seed_{seed}.png"
                plt.savefig(output_path, dpi=200)
                plt.close()
                figure_paths.append(str(output_path))
    return figure_paths
