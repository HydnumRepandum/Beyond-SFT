from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .config import canonical_label_ids
from .modeling import model_device, prepare_for_inference
from .prompting import candidate_responses

# Keep Torch behind modeling so modeling's Unsloth import happens before
# transformer/PEFT-related imports in notebook inference sessions.
import torch
import torch.nn.functional as F


def _score_single_prompt_candidates(model, tokenizer, prompt: str, config: dict[str, Any]) -> dict[str, float]:
    response_map = candidate_responses(config)
    labels = list(response_map.keys())
    full_texts = [prompt + response_map[label] for label in labels]

    prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    batch = tokenizer(
        full_texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=int(config["model"]["max_seq_length"]),
        return_tensors="pt",
    )
    device = model_device(model)
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, :-1, :].contiguous()

    shifted_labels = batch["input_ids"][:, 1:].clone()
    shifted_labels[batch["attention_mask"][:, 1:] == 0] = -100
    shifted_labels[:, : max(prompt_len - 1, 0)] = -100

    token_losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(logits.size(0), -1)

    valid_mask = shifted_labels.ne(-100)
    lengths = valid_mask.sum(dim=1).clamp_min(1)
    mean_log_probs = -(token_losses * valid_mask).sum(dim=1) / lengths
    return {label: float(score) for label, score in zip(labels, mean_log_probs.tolist())}


def score_dataframe(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    progress_label: str,
) -> pd.DataFrame:
    model = prepare_for_inference(model)
    rows: list[dict[str, Any]] = []
    label_ids = canonical_label_ids(config)

    for row in tqdm(dataframe.to_dict("records"), desc=progress_label):
        prompt = row["prompt"]
        scores = _score_single_prompt_candidates(model, tokenizer, prompt, config)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predicted_label = ranked[0][0]
        score_margin = ranked[0][1] - ranked[1][1]
        enriched = dict(row)
        for label_id in label_ids:
            enriched[f"score__{label_id}"] = scores[label_id]
        enriched["predicted_label"] = predicted_label
        enriched["predicted_response"] = candidate_responses(config)[predicted_label]
        enriched["score_margin"] = score_margin
        enriched["hard_negative_label"] = next(label for label, _ in ranked if label != row.get("label"))
        enriched["is_correct"] = bool(row.get("label") == predicted_label) if "label" in row else None
        rows.append(enriched)
    return pd.DataFrame(rows)


def _existing_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for column in columns:
        if column in dataframe.columns and column not in seen:
            selected.append(column)
            seen.add(column)
    return selected


def logprob_prediction_table(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    score_columns = [f"score__{label_id}" for label_id in canonical_label_ids(config)]
    ordered_columns = [
        "example_id",
        "source_example_id",
        "source_row_index",
        "source_split",
        "timestamp",
        "text",
        "label_raw",
        "label_raw_normalized",
        "label",
        "prompt",
        "predicted_label",
        "predicted_response",
        "score_margin",
        "hard_negative_label",
        "is_correct",
        "is_hard_primary",
        "is_hard_secondary",
        *score_columns,
    ]
    return predictions.loc[:, _existing_columns(predictions, ordered_columns)].copy()


def write_prediction_artifacts(
    predictions: pd.DataFrame,
    run_dir: str | Path,
    split_name: str,
    config: dict[str, Any],
) -> None:
    run_path = Path(run_dir)
    predictions.to_csv(run_path / f"predictions_{split_name}.csv", index=False)

    inference_cfg = config.get("inference", {})
    if bool(inference_cfg.get("save_logprob_predictions", True)):
        logprob_prediction_table(predictions, config).to_csv(
            run_path / f"logprob_predictions_{split_name}.csv",
            index=False,
        )
