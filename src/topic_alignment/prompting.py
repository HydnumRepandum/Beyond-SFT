from __future__ import annotations

import json
from typing import Any

from .config import canonical_label_ids, task_instruction_text, task_labels


def canonical_response(label_id: str) -> str:
    return json.dumps({"label": label_id}, separators=(",", ":"))


def canonical_instruction(config: dict[str, Any]) -> str:
    instruction_text = task_instruction_text(config)
    if instruction_text:
        base_instruction = instruction_text.strip()
    else:
        label_lines = [f"- {label['id']}: {label['description']}" for label in task_labels(config)]
        labels_block = "\n".join(label_lines)
        base_instruction = (
            "Classify the political text into exactly one label from the closed label set below.\n\n"
            "Label codebook:\n"
            f"{labels_block}"
        )

    allowed_labels = ", ".join(canonical_label_ids(config))
    return (
        f"{base_instruction}\n\n"
        "Return exactly one minified JSON object and nothing else.\n"
        f'Allowed labels: {allowed_labels}\n'
        'Required schema: {"label":"<ONE_LABEL>"}'
    )


def build_prompt(text: str, config: dict[str, Any]) -> str:
    clean_text = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    instruction = canonical_instruction(config)
    if "{text}" in instruction:
        prompt_body = instruction.replace("{text}", clean_text)
    else:
        prompt_body = f"{instruction}\n\nText:\n{clean_text}"
    return f"{prompt_body}{config['format']['response_prefix']}"


def build_sft_text(text: str, label_id: str, config: dict[str, Any]) -> str:
    return build_prompt(text, config) + canonical_response(label_id)


def candidate_responses(config: dict[str, Any]) -> dict[str, str]:
    return {label_id: canonical_response(label_id) for label_id in canonical_label_ids(config)}


def build_preference_row(text: str, chosen_label: str, rejected_label: str, config: dict[str, Any]) -> dict[str, str]:
    prompt = build_prompt(text, config)
    return {
        "prompt": prompt,
        "chosen": canonical_response(chosen_label),
        "rejected": canonical_response(rejected_label),
    }


def build_kto_rows(text: str, gold_label: str, config: dict[str, Any]) -> list[dict[str, str | bool]]:
    prompt = build_prompt(text, config)
    rows: list[dict[str, str | bool]] = []
    for label_id in canonical_label_ids(config):
        rows.append(
            {
                "prompt": prompt,
                "completion": canonical_response(label_id),
                "label": label_id == gold_label,
            }
        )
    return rows
