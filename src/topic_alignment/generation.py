"""Free-generation classification, for comparison with candidate log-prob scoring.

The same model, prompt and test items are used as in `inference.score_dataframe`; only
the way a label is read out of the model changes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import canonical_label_ids
from .prompting import canonical_response

INVALID = "__INVALID__"


def parse_strict(output: str, labels: list[str]) -> str:
    """Accept only the exact canonical JSON object, optionally followed by more text."""
    text = output.strip()
    for label in labels:
        if text.startswith(canonical_response(label)):
            return label
    return INVALID


def parse_lenient(output: str, labels: list[str]) -> str:
    """Recover a label from loosely formatted output, as a human post-processor would.

    Tries a `label: X` pattern first, then the first standalone label token.
    """
    alternatives = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    keyed = re.search(rf"label[\"']?\s*[:=]\s*[\"']?({alternatives})\b", output)
    if keyed:
        return keyed.group(1)
    standalone = re.search(rf"(?<![A-Za-z0-9])({alternatives})(?![A-Za-z0-9])", output)
    return standalone.group(1) if standalone else INVALID


def majority_vote(votes: list[str]) -> str:
    """Most frequent valid label; ties go to the label that was generated first."""
    valid = [vote for vote in votes if vote != INVALID]
    if not valid:
        return INVALID
    counts = Counter(valid)
    best = max(counts.values())
    return next(vote for vote in valid if counts[vote] == best)


def generate_dataframe(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    num_samples: int = 1,
    temperature: float = 1.0,
    max_new_tokens: int = 12,
    progress_label: str = "generation",
) -> pd.DataFrame:
    """Generate a completion per prompt.

    mode="greedy" takes the single most likely continuation. mode="sample" draws
    `num_samples` continuations at `temperature`, which is what repeated API calls do.
    """
    import torch

    from .modeling import model_device, prepare_for_inference

    if mode not in {"greedy", "sample"}:
        raise ValueError(f"Unsupported generation mode: {mode}")
    model = prepare_for_inference(model)
    labels = canonical_label_ids(config)
    device = model_device(model)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []

    for row in tqdm(dataframe.to_dict("records"), desc=progress_label):
        encoded = tokenizer(
            row["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=int(config["model"]["max_seq_length"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_token_id}
        if mode == "greedy":
            generation_kwargs.update(do_sample=False)
        else:
            generation_kwargs.update(
                do_sample=True,
                temperature=float(temperature),
                top_p=1.0,
                top_k=0,
                num_return_sequences=int(num_samples),
            )
        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        outputs = tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)

        strict = [parse_strict(output, labels) for output in outputs]
        lenient = [parse_lenient(output, labels) for output in outputs]
        rows.append(
            {
                "example_id": row["example_id"],
                "label": row["label"],
                "raw_outputs": json.dumps(outputs, ensure_ascii=True),
                "strict_labels": json.dumps(strict),
                "lenient_labels": json.dumps(lenient),
                "predicted_label_strict": majority_vote(strict),
                # `predicted_label` is the lenient majority vote, the most favourable
                # reading of the generated text.
                "predicted_label": majority_vote(lenient),
            }
        )
    return pd.DataFrame(rows)


def generation_metrics(generated: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Validity, per-sample accuracy spread and run-to-run agreement of generated labels."""
    from .evaluation import compute_metrics

    strict = np.array([json.loads(value) for value in generated["strict_labels"]])
    lenient = np.array([json.loads(value) for value in generated["lenient_labels"]])
    gold = generated["label"].to_numpy()[:, None]
    sample_accuracy = (lenient == gold).mean(axis=0)

    sample_count = lenient.shape[1]
    agreement = None
    if sample_count > 1:
        pairs = [
            float((lenient[:, first] == lenient[:, second]).mean())
            for first in range(sample_count)
            for second in range(first + 1, sample_count)
        ]
        agreement = float(np.mean(pairs))

    def scored(column: str) -> dict[str, Any]:
        metrics = compute_metrics(generated.assign(predicted_label=generated[column]), config)
        return {key: metrics[key] for key in ["accuracy", "macro_f1", "invalid_prediction_count"]}

    return {
        "sample_count": int(sample_count),
        "invalid_rate_strict": float((strict == INVALID).mean()),
        "invalid_rate_lenient": float((lenient == INVALID).mean()),
        # Invalid outputs count as errors in every accuracy below.
        "single_sample_accuracy_mean": float(sample_accuracy.mean()),
        "single_sample_accuracy_std": float(sample_accuracy.std(ddof=1)) if sample_count > 1 else None,
        "pairwise_sample_agreement": agreement,
        "majority_vote_lenient": scored("predicted_label"),
        "majority_vote_strict": scored("predicted_label_strict"),
    }
