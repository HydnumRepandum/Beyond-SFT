"""Classify the test set by free generation and compare it with log-prob scoring.

Run it on a stage that already has `predictions_test.csv` (log-prob scoring), e.g.

    python scripts/score_generation.py --run baseline --mode greedy
    python scripts/score_generation.py --run seed_11/train_100/allocation_shared/sft --mode sample --num-samples 5
"""

from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401

from topic_alignment.config import load_config
from topic_alignment.data import load_prepared_split
from topic_alignment.generation import generate_dataframe, generation_metrics
from topic_alignment.modeling import load_base_model, load_model_with_adapter
from topic_alignment.stats import mcnemar_exact
from topic_alignment.utils import resolve_path, set_global_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/llama31_revision.yaml")
    parser.add_argument("--task", default=None, help="Task key from the config task_registry.")
    parser.add_argument("--run", required=True, help="Run directory relative to <outputs_root>/runs.")
    parser.add_argument("--mode", required=True, choices=["greedy", "sample"])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    config = load_config(args.config, task_key=args.task)
    run_dir = resolve_path(config["project"]["outputs_root"]) / "runs" / args.run
    logprob_path = run_dir / "predictions_test.csv"
    if not logprob_path.exists():
        raise RuntimeError(f"Missing {logprob_path}. Score this run with log-probs first.")

    adapter_dir = run_dir / "adapter"
    if adapter_dir.exists():
        model, tokenizer = load_model_with_adapter(config, adapter_dir, trainable=False)
    else:
        model, tokenizer = load_base_model(config)

    set_global_seed(args.seed)
    test_df = load_prepared_split(config, "test")
    generated = generate_dataframe(
        model,
        tokenizer,
        test_df,
        config,
        mode=args.mode,
        num_samples=1 if args.mode == "greedy" else args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        progress_label=f"generation-{args.mode}",
    )
    tag = "greedy" if args.mode == "greedy" else f"sample_t{args.temperature:g}_n{args.num_samples}"
    generated.to_csv(run_dir / f"generation_{tag}.csv", index=False)

    logprob = pd.read_csv(logprob_path).set_index("example_id").reindex(generated["example_id"])
    logprob_correct = (logprob["predicted_label"] == logprob["label"]).to_numpy()
    generated_correct = (generated["predicted_label"] == generated["label"]).to_numpy()
    payload = {
        "run": args.run,
        "mode": args.mode,
        "temperature": None if args.mode == "greedy" else args.temperature,
        "generation": generation_metrics(generated, config),
        "logprob_accuracy": float(logprob_correct.mean()),
        "label_agreement_with_logprob": float(
            (logprob["predicted_label"].to_numpy() == generated["predicted_label"].to_numpy()).mean()
        ),
        # a = log-prob scoring, b = generation (lenient majority vote), same test items.
        "mcnemar_logprob_vs_generation": mcnemar_exact(logprob_correct, generated_correct),
    }
    write_json(run_dir / f"generation_{tag}_metrics.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
