from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .utils import repo_root, resolve_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_repo_root"] = str(repo_root())
    config["_config_hash"] = config_hash(config)
    return config


def _public_config_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_config_payload(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_public_config_payload(item) for item in value]
    return value


def config_hash(config: dict[str, Any]) -> str:
    payload = _public_config_payload(config)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _clean_mapping_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "Ò": '"',
        "Ó": '"',
        "Õ": "'",
        "Ñ": "-",
        "Ð": "-",
        "\r\n": "\n",
        "\r": "\n",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.strip()


def task_metadata(config: dict[str, Any]) -> dict[str, Any]:
    cached = config.get("_task_metadata")
    if cached is not None:
        return cached

    task_cfg = config["task"]
    metadata: dict[str, Any] = {}
    mapping_path = task_cfg.get("mapping_path")
    if mapping_path:
        mapping_df = pd.read_csv(
            resolve_path(mapping_path),
            sep=str(task_cfg.get("mapping_delimiter", ";")),
            encoding=str(task_cfg.get("mapping_encoding", "cp1252")),
            engine="python",
        )
        dataset_number = int(task_cfg["mapping_dataset_number"])
        task_number = int(task_cfg["mapping_task_number"])
        matched = mapping_df[
            (mapping_df["dataset_number"].astype(int) == dataset_number)
            & (mapping_df["task_number"].astype(int) == task_number)
        ]
        if matched.empty:
            raise ValueError(
                f"Could not find mapping row for dataset_number={dataset_number}, task_number={task_number}"
            )
        row = matched.iloc[0]
        label_ids = [item.strip() for item in str(row["labelset"]).split(",") if item.strip()]
        label_words = [item.strip() for item in str(row["labelset_fullword"]).split(";") if item.strip()]
        if label_words and len(label_ids) != len(label_words):
            raise ValueError(
                "Mapping file labelset and labelset_fullword lengths do not match: "
                f"{len(label_ids)} vs {len(label_words)}"
            )
        metadata = {
            "dataset_filename": str(row["dataset_filename"]).strip(),
            "zero_shot_prompt": _clean_mapping_text(row["zero_shot_prompt"]),
            "label_column": str(row.get("label_column", "")).strip() or None,
            "labels": [
                {
                    "id": label_id,
                    "raw_values": [label_word.lower()],
                    "display_name": label_word.title(),
                    "description": label_word.title(),
                }
                for label_id, label_word in zip(label_ids, label_words or label_ids)
            ],
        }

    config["_task_metadata"] = metadata
    return metadata


def task_labels(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config["task"].get("labels")
    if configured:
        return list(configured)
    derived = task_metadata(config).get("labels")
    if derived:
        return list(derived)
    raise ValueError("No task labels configured and no labels could be derived from task metadata.")


def canonical_label_ids(config: dict[str, Any]) -> list[str]:
    return [label["id"] for label in task_labels(config)]


def raw_to_canonical_map(config: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in task_labels(config):
        for raw_value in label["raw_values"]:
            mapping[raw_value.strip().lower()] = label["id"]
    return mapping


def canonical_to_display_map(config: dict[str, Any]) -> dict[str, str]:
    return {label["id"]: label["display_name"] for label in task_labels(config)}


def canonical_to_description_map(config: dict[str, Any]) -> dict[str, str]:
    return {label["id"]: label["description"] for label in task_labels(config)}


def task_instruction_text(config: dict[str, Any]) -> str | None:
    format_cfg = config.get("format", {})
    if format_cfg.get("instruction_text"):
        return str(format_cfg["instruction_text"]).strip()
    return task_metadata(config).get("zero_shot_prompt")


def reference_train_size(config: dict[str, Any], available_train_sizes: list[int | str] | None = None) -> int | str:
    configured = config["task"].get("reference_train_size")
    if configured is not None:
        return configured
    if available_train_sizes is None:
        return "full"
    if "full" in available_train_sizes:
        return "full"
    numeric_sizes = [int(size) for size in available_train_sizes if str(size).isdigit()]
    if not numeric_sizes:
        raise ValueError("Could not determine a reference train size from the available train sizes.")
    return max(numeric_sizes)


def data_allocation_variants(config: dict[str, Any]) -> list[str]:
    variants = config.get("dpo", {}).get("data_allocation_variants")
    if not variants:
        return ["shared"]
    return [str(variant) for variant in variants]
