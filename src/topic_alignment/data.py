from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import canonical_label_ids, raw_to_canonical_map, reference_train_size, task_label_column
from .prompting import build_prompt, build_sft_text, canonical_response
from .utils import ensure_dir, resolve_path, set_global_seed, write_json, write_text


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def _normalize_label(value: Any) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _task_mode(config: dict[str, Any]) -> str:
    return str(config["task"].get("data_mode", "raw_single_csv"))


def _standardize_columns(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    renamed = dataframe.rename(
        columns={
            config["task"]["id_column"]: "example_id",
            config["task"].get("timestamp_column", ""): "timestamp",
            config["task"]["text_column"]: "text_raw",
            task_label_column(config): "label_raw",
        }
    ).copy()
    if "timestamp" not in renamed.columns:
        renamed["timestamp"] = None
    renamed["source_example_id"] = renamed["example_id"]
    normalized_ids = renamed["source_example_id"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    duplicate_ids = normalized_ids.duplicated(keep=False) | normalized_ids.eq("")
    renamed["example_id"] = normalized_ids
    renamed.loc[duplicate_ids, "example_id"] = (
        normalized_ids[duplicate_ids] + "__row_" + renamed.index[duplicate_ids].astype(str)
    )
    renamed["text"] = renamed["text_raw"].map(_normalize_text)
    renamed["label_raw_normalized"] = renamed["label_raw"].map(_normalize_label)
    renamed["source_row_index"] = renamed.index
    return renamed


def _load_external_csv(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    return _standardize_columns(dataframe, config)


def _provided_train_sizes(config: dict[str, Any]) -> list[int]:
    task_cfg = config["task"]
    explicit = task_cfg.get("provided_train_sizes")
    if explicit:
        return [int(size) for size in explicit]
    return [int(size) for size in task_cfg["requested_train_sizes"] if str(size).isdigit()]


def _full_train_source_size(config: dict[str, Any]) -> int:
    task_cfg = config["task"]
    if task_cfg.get("full_train_source") is not None:
        return int(task_cfg["full_train_source"])
    return max(_provided_train_sizes(config))


def _external_train_subset_path(config: dict[str, Any], train_size: int) -> Path:
    task_cfg = config["task"]
    filename = str(task_cfg["external_train_pattern"]).format(train_size=train_size)
    return resolve_path(task_cfg["samples_dir"]) / filename


def _external_test_path(config: dict[str, Any]) -> Path:
    task_cfg = config["task"]
    filename = str(task_cfg["external_test_filename"])
    return resolve_path(task_cfg["samples_dir"]) / filename


def load_raw_task_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    if _task_mode(config) == "provided_subsets":
        reference_train = _load_external_csv(_external_train_subset_path(config, _full_train_source_size(config)), config)
        reference_train["source_split"] = "provided_train_reference"
        test_df = _load_external_csv(_external_test_path(config), config)
        test_df["source_split"] = "provided_test"
        return pd.concat([reference_train, test_df], ignore_index=True)

    dataset_path = resolve_path(config["task"]["raw_dataset_path"])
    return _standardize_columns(pd.read_csv(dataset_path), config)


def validate_raw_task_dataframe(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    raw_label_mapping = raw_to_canonical_map(config)
    source_id_series = dataframe.get("source_example_id", dataframe["example_id"])
    invalid_labels = sorted(
        {
            label
            for label in dataframe["label_raw_normalized"].dropna().unique().tolist()
            if label not in raw_label_mapping
        }
    )
    summary = {
        "row_count_total": int(len(dataframe)),
        "missing_text_rows": int((dataframe["text"] == "").sum()),
        "missing_label_rows": int(dataframe["label_raw_normalized"].isna().sum()),
        "duplicate_example_id_rows": int(source_id_series.duplicated().sum()),
        "duplicate_text_rows": int(dataframe["text"].duplicated().sum()),
        "invalid_labels": invalid_labels,
        "raw_label_counts": {
            str(key): int(value)
            for key, value in Counter(dataframe["label_raw_normalized"].fillna("MISSING")).items()
        },
    }
    if _task_mode(config) == "provided_subsets":
        summary["provided_train_sizes"] = _provided_train_sizes(config)
        summary["full_train_source"] = _full_train_source_size(config)
        summary["provided_test_rows"] = int(len(dataframe[dataframe.get("source_split") == "provided_test"]))
    return summary


def build_labeled_task_dataframe(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labeled = dataframe.copy()
    label_mapping = raw_to_canonical_map(config)
    labeled["label"] = labeled["label_raw_normalized"].map(label_mapping)

    if config["task"].get("drop_missing_labels", True):
        labeled = labeled[labeled["label"].notna()].copy()

    labeled["gold_response"] = labeled["label"].map(canonical_response)
    labeled["prompt"] = labeled["text"].map(lambda text: build_prompt(text, config))
    labeled["sft_text"] = labeled.apply(
        lambda row: build_sft_text(row["text"], row["label"], config),
        axis=1,
    )
    labeled["label_display"] = labeled["label"]
    labeled = labeled.reset_index(drop=True)
    return labeled


def _make_raw_single_csv_splits(dataframe: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    seed = int(config["task"]["split_seed"])
    test_size = int(config["task"]["test_size"])
    label_col = "label"

    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=dataframe[label_col],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def _make_provided_subset_splits(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[int, pd.DataFrame]]:
    test_df = build_labeled_task_dataframe(_load_external_csv(_external_test_path(config), config), config)

    subset_map: dict[int, pd.DataFrame] = {}
    for train_size in _provided_train_sizes(config):
        subset_map[train_size] = build_labeled_task_dataframe(
            _load_external_csv(_external_train_subset_path(config, train_size), config),
            config,
        )

    full_reference = subset_map[_full_train_source_size(config)].copy().reset_index(drop=True)
    split_map = {
        "train": full_reference,
        "test": test_df.reset_index(drop=True),
    }
    return split_map, subset_map


def make_splits_and_subsets(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[int | str, pd.DataFrame]]:
    if _task_mode(config) == "provided_subsets":
        split_map, provided_subsets = _make_provided_subset_splits(config)
        subsets: dict[int | str, pd.DataFrame] = {int(size): df for size, df in provided_subsets.items()}
        subsets["full"] = provided_subsets[_full_train_source_size(config)].copy().reset_index(drop=True)
    else:
        labeled_df = build_labeled_task_dataframe(load_raw_task_dataframe(config), config)
        split_map = _make_raw_single_csv_splits(labeled_df, config)
        subsets = {}
        for requested in compute_available_train_sizes(split_map["train"], config):
            subsets[requested] = _select_subset(split_map["train"], requested, int(config["task"]["split_seed"]))
    expected_labels = set(canonical_label_ids(config))
    for split_name, split_df in split_map.items():
        observed = set(split_df["label"].unique().tolist())
        missing = expected_labels - observed
        if missing:
            raise ValueError(f"Split '{split_name}' is missing labels: {sorted(missing)}")
    return split_map, subsets


def _select_subset(train_df: pd.DataFrame, train_size: int | str, seed: int) -> pd.DataFrame:
    if train_size == "full":
        return train_df.copy().reset_index(drop=True)
    train_size_int = int(train_size)
    if train_size_int > len(train_df):
        raise ValueError(f"Requested train size {train_size_int} exceeds available train rows {len(train_df)}")
    if train_size_int == len(train_df):
        return train_df.copy().reset_index(drop=True)
    labels = sorted(train_df["label"].unique().tolist())
    if train_size_int < len(labels):
        raise ValueError(f"Requested train size {train_size_int} is smaller than the number of classes {len(labels)}")

    total_rows = len(train_df)
    class_counts = train_df["label"].value_counts().to_dict()
    raw_targets = {label: train_size_int * class_counts[label] / total_rows for label in labels}
    target_counts = {
        label: min(class_counts[label], max(1, math.floor(raw_targets[label])))
        for label in labels
    }

    while sum(target_counts.values()) > train_size_int:
        removable = [label for label in labels if target_counts[label] > 1]
        if not removable:
            raise ValueError("Could not reduce stratified subset counts to requested size.")
        label_to_reduce = min(
            removable,
            key=lambda label: (raw_targets[label] - target_counts[label], target_counts[label]),
        )
        target_counts[label_to_reduce] -= 1

    while sum(target_counts.values()) < train_size_int:
        expandable = [label for label in labels if target_counts[label] < class_counts[label]]
        if not expandable:
            raise ValueError("Could not expand stratified subset counts to requested size.")
        label_to_expand = max(
            expandable,
            key=lambda label: (raw_targets[label] - target_counts[label], -target_counts[label]),
        )
        target_counts[label_to_expand] += 1

    sampled_frames = []
    for offset, label in enumerate(labels):
        label_df = train_df[train_df["label"] == label]
        sampled_frames.append(
            label_df.sample(
                n=target_counts[label],
                random_state=seed + offset,
                replace=False,
            )
        )

    subset_df = pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset_df


def allocate_training_rows(
    train_df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    data_allocation: str,
) -> dict[str, Any]:
    allocation = str(data_allocation)
    full_train = train_df.copy().reset_index(drop=True)
    if allocation == "shared":
        return {
            "allocation": allocation,
            "sft_train": full_train,
            "preference_source": full_train.copy().reset_index(drop=True),
            "summary": {
                "allocation": allocation,
                "source_row_count": int(len(full_train)),
                "sft_train_row_count": int(len(full_train)),
                "preference_source_row_count": int(len(full_train)),
            },
        }

    if allocation != "sft25_dpo75":
        raise ValueError(f"Unsupported data allocation: {allocation}")

    sft_fraction = float(config["dpo"].get("split_allocation_sft_fraction", 0.25))
    if not 0.0 < sft_fraction < 1.0:
        raise ValueError(f"split_allocation_sft_fraction must be between 0 and 1, got {sft_fraction}")
    dpo_fraction = 1.0 - sft_fraction

    sft_train_df, preference_source_df = train_test_split(
        full_train,
        test_size=dpo_fraction,
        random_state=seed,
        stratify=full_train["label"],
    )
    sft_train_df = sft_train_df.reset_index(drop=True)
    preference_source_df = preference_source_df.reset_index(drop=True)

    expected_labels = set(canonical_label_ids(config))
    for split_name, split_df in [("sft_train", sft_train_df), ("preference_source", preference_source_df)]:
        missing = expected_labels - set(split_df["label"].unique().tolist())
        if missing:
            raise ValueError(
                f"Allocation '{allocation}' produced a {split_name} split missing labels: {sorted(missing)}"
            )

    return {
        "allocation": allocation,
        "sft_train": sft_train_df,
        "preference_source": preference_source_df,
        "summary": {
            "allocation": allocation,
            "source_row_count": int(len(full_train)),
            "sft_train_row_count": int(len(sft_train_df)),
            "preference_source_row_count": int(len(preference_source_df)),
            "sft_fraction": sft_fraction,
            "preference_fraction": dpo_fraction,
        },
    }


def compute_available_train_sizes(train_df: pd.DataFrame, config: dict[str, Any]) -> list[int | str]:
    if _task_mode(config) == "provided_subsets":
        available: list[int | str] = []
        available_sizes = set(_provided_train_sizes(config))
        for requested in config["task"]["requested_train_sizes"]:
            if requested == "full":
                available.append("full")
            elif int(requested) in available_sizes:
                available.append(int(requested))
        return available

    available = []
    for requested in config["task"]["requested_train_sizes"]:
        if requested == "full":
            available.append("full")
            continue
        if int(requested) <= len(train_df):
            available.append(int(requested))
    return available


def dataset_summary_markdown(
    raw_summary: dict[str, Any],
    labeled_df: pd.DataFrame,
    split_map: dict[str, pd.DataFrame],
    available_train_sizes: list[int | str],
) -> str:
    class_counts = labeled_df["label"].value_counts().sort_index()
    split_counts = {split: len(df) for split, df in split_map.items()}
    lines = [
        "# Dataset Summary",
        "",
        "## Raw audit",
        f"- Total rows inspected: {raw_summary['row_count_total']}",
        f"- Missing text rows: {raw_summary['missing_text_rows']}",
        f"- Missing label rows: {raw_summary['missing_label_rows']}",
        f"- Duplicate text rows: {raw_summary['duplicate_text_rows']}",
        f"- Duplicate example-id rows: {raw_summary['duplicate_example_id_rows']}",
        f"- Invalid labels: {raw_summary['invalid_labels'] or 'None'}",
        "",
        "## Labeled task rows",
        f"- Labeled rows kept for the study: {len(labeled_df)}",
        "",
        "## Class counts",
    ]
    for label, count in class_counts.items():
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Deterministic split sizes",
            f"- Train reference split: {split_counts['train']}",
        ]
    )
    lines.append(f"- Test: {split_counts['test']}")
    lines.extend(
        [
            "",
            f"## Available train sizes: {available_train_sizes}",
        ]
    )
    if raw_summary.get("provided_train_sizes"):
        lines.extend(
            [
                "",
                "## Provided subset mode notes",
                f"- Provided train subset sizes: {raw_summary['provided_train_sizes']}",
                f"- The train reference split is loaded from the provided {raw_summary['full_train_source']}-example training file.",
                "- The provided held-out file is kept intact as the test split.",
            ]
        )
    return "\n".join(lines) + "\n"


def _manifest_path(config: dict[str, Any], path: Path) -> str:
    repo_root = resolve_path(config.get("_repo_root", "."))
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def prepare_dataset_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    set_global_seed(int(config["task"]["split_seed"]))
    raw_df = load_raw_task_dataframe(config)
    raw_summary = validate_raw_task_dataframe(raw_df, config)
    split_map, prepared_subsets = make_splits_and_subsets(config)
    available_train_sizes = compute_available_train_sizes(split_map["train"], config)
    labeled_df = pd.concat([split_df for split_df in split_map.values()], ignore_index=True)

    data_root = ensure_dir(resolve_path(config["project"]["data_root"]))
    processed_dir = ensure_dir(data_root / config["task"]["name"])
    splits_dir = ensure_dir(processed_dir / "splits")
    subsets_dir = ensure_dir(processed_dir / "subsets")

    labeled_path = processed_dir / "labeled_examples.csv"
    labeled_df.to_csv(labeled_path, index=False)

    for split_name, split_df in split_map.items():
        split_df.to_csv(splits_dir / f"{split_name}.csv", index=False)

    for seed in config["task"]["subset_seeds"]:
        seed_dir = ensure_dir(subsets_dir / f"seed_{seed}")
        for train_size in available_train_sizes:
            subset_df = prepared_subsets[train_size].copy().reset_index(drop=True)
            size_tag = str(train_size).lower()
            subset_df.to_csv(seed_dir / f"train_{size_tag}.csv", index=False)

    summary_payload = {
        "raw_summary": raw_summary,
        "labeled_row_count": int(len(labeled_df)),
        "class_counts": labeled_df["label"].value_counts().sort_index().to_dict(),
        "split_sizes": {split_name: int(len(split_df)) for split_name, split_df in split_map.items()},
        "available_train_sizes": available_train_sizes,
        "notes": {
            "data_mode": _task_mode(config),
            "full_train_reference": reference_train_size(config, available_train_sizes),
        },
    }
    summary_path = processed_dir / "dataset_summary.json"
    write_json(summary_path, summary_payload)
    write_text(
        processed_dir / "dataset_summary.md",
        dataset_summary_markdown(raw_summary, labeled_df, split_map, available_train_sizes),
    )

    manifest = {
        "processed_dir": _manifest_path(config, processed_dir),
        "labeled_path": _manifest_path(config, labeled_path),
        "splits_dir": _manifest_path(config, splits_dir),
        "subsets_dir": _manifest_path(config, subsets_dir),
        "summary_path": _manifest_path(config, summary_path),
        "available_train_sizes": available_train_sizes,
    }
    write_json(processed_dir / "manifest.json", manifest)
    return manifest


def processed_task_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config["project"]["data_root"]) / config["task"]["name"]


def load_prepared_split(config: dict[str, Any], split_name: str) -> pd.DataFrame:
    split_path = processed_task_dir(config) / "splits" / f"{split_name}.csv"
    if not split_path.exists():
        prepare_dataset_artifacts(config)
    return pd.read_csv(split_path)


def load_prepared_subset(config: dict[str, Any], subset_seed: int, train_size: int | str) -> pd.DataFrame:
    size_tag = str(train_size).lower()
    subset_path = processed_task_dir(config) / "subsets" / f"seed_{subset_seed}" / f"train_{size_tag}.csv"
    if not subset_path.exists():
        prepare_dataset_artifacts(config)
    return pd.read_csv(subset_path)
