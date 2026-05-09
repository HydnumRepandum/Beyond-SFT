from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .utils import repo_root, resolve_path

PROMPT_COLUMNS = [
    "zero_shot_prompt",
    "few_shot_prompt",
    "few_shot_chain_of_thought_prompt",
]

PROMPT_SOURCE_ALIASES = {
    "zero_shot": "zero_shot_prompt",
    "few_shot": "few_shot_prompt",
    "few_shot_cot": "few_shot_chain_of_thought_prompt",
    "chain_of_thought": "few_shot_chain_of_thought_prompt",
}

def load_config(config_path: str | Path, task_key: str | None = None) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_repo_root"] = str(repo_root())
    _apply_task_selection(config, task_key=task_key)
    config["_config_hash"] = config_hash(config)
    return config


def _apply_task_selection(config: dict[str, Any], task_key: str | None = None) -> None:
    task_cfg = dict(config.get("task", {}))
    registry = task_cfg.get("task_registry") or {}
    selected_task = task_key or task_cfg.get("selected_task")
    if not registry:
        if task_key is not None:
            raise ValueError("This config does not define task_registry, so --task cannot be used.")
        return
    if not selected_task:
        return
    if selected_task not in registry:
        raise ValueError(
            f"Unknown task '{selected_task}'. Available tasks: {', '.join(sorted(registry))}"
        )

    base_task = {
        key: value
        for key, value in task_cfg.items()
        if key not in {"task_registry", "selected_task"}
    }
    selected_task_cfg = dict(registry[selected_task])
    merged_task = {**base_task, **selected_task_cfg}
    merged_task["selected_task"] = selected_task
    merged_task["available_tasks"] = sorted(registry)
    config["task"] = merged_task
    _apply_task_dependent_kto_weights(config, selected_task)

    project_cfg = config.setdefault("project", {})
    if project_cfg.get("study_name") and selected_task not in str(project_cfg["study_name"]):
        project_cfg["study_name"] = f"{project_cfg['study_name']}_{selected_task}"
    if bool(project_cfg.get("task_scoped_outputs", True)):
        for root_key in ["outputs_root", "results_root"]:
            root_value = project_cfg.get(root_key)
            if root_value is None:
                continue
            root = Path(str(root_value))
            if root.name != selected_task:
                project_cfg[root_key] = str(root / selected_task)

    tracking_cfg = config.get("tracking", {})
    if tracking_cfg.get("project_name") and selected_task not in str(tracking_cfg["project_name"]):
        tracking_cfg["project_name"] = f"{tracking_cfg['project_name']}_{selected_task}"

    format_cfg = config.get("format", {})
    if format_cfg.get("prompt_version") and selected_task not in str(format_cfg["prompt_version"]):
        prompt_source = str(format_cfg.get("prompt_source", "zero_shot_prompt"))
        format_cfg["prompt_version"] = f"{selected_task}_{prompt_source}"


def _apply_task_dependent_kto_weights(config: dict[str, Any], selected_task: str) -> None:
    kto_cfg = config.get("kto")
    if not isinstance(kto_cfg, dict):
        return
    desirable_weight_by_task = dict(kto_cfg.get("desirable_weight_by_task") or {})
    if selected_task in desirable_weight_by_task:
        kto_cfg["desirable_weight"] = float(desirable_weight_by_task[selected_task])


def available_task_keys(config: dict[str, Any]) -> list[str]:
    task_cfg = config.get("task", {})
    if task_cfg.get("available_tasks"):
        return list(task_cfg["available_tasks"])
    registry = task_cfg.get("task_registry") or {}
    return sorted(registry)


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
    if value is None or pd.isna(value):
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
        prompts = {
            column: _clean_mapping_text(row[column])
            for column in PROMPT_COLUMNS
            if column in row and _clean_mapping_text(row[column])
        }
        metadata = {
            "dataset_filename": str(row["dataset_filename"]).strip(),
            "prompts": prompts,
            "zero_shot_prompt": prompts.get("zero_shot_prompt", ""),
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
        return _apply_label_aliases(config, list(configured))
    derived = task_metadata(config).get("labels")
    if derived:
        return _apply_label_aliases(config, list(derived))
    raise ValueError("No task labels configured and no labels could be derived from task metadata.")


def _apply_label_aliases(config: dict[str, Any], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases = config.get("task", {}).get("label_aliases") or {}
    normalized_labels: list[dict[str, Any]] = []
    for label in labels:
        copied = dict(label)
        raw_values = list(copied.get("raw_values", []))
        for alias in aliases.get(copied["id"], []):
            alias_text = str(alias)
            if alias_text not in raw_values:
                raw_values.append(alias_text)
        copied["raw_values"] = raw_values
        normalized_labels.append(copied)
    return normalized_labels


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
    metadata = task_metadata(config)
    prompts = metadata.get("prompts", {})
    prompt_source = str(format_cfg.get("prompt_source", "zero_shot_prompt"))
    prompt_column = PROMPT_SOURCE_ALIASES.get(prompt_source, prompt_source)
    if prompt_column in prompts:
        return prompts[prompt_column]
    if prompts:
        raise ValueError(
            f"Prompt source '{prompt_source}' is not available for this task. "
            f"Available prompt sources: {', '.join(sorted(prompts))}"
        )
    return metadata.get("zero_shot_prompt")


def task_label_column(config: dict[str, Any]) -> str:
    configured = config["task"].get("raw_label_column")
    if configured:
        return str(configured)
    derived = task_metadata(config).get("label_column")
    if derived:
        return str(derived)
    raise ValueError("No raw_label_column configured and no label_column could be derived from task metadata.")


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
