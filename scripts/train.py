from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401

from topic_alignment.config import canonical_label_ids, data_allocation_variants, load_config
from topic_alignment.data import (
    allocate_training_rows,
    load_prepared_split,
    load_prepared_subset,
    prepare_dataset_artifacts,
)
from topic_alignment.evaluation import compute_metrics_bundle
from topic_alignment.inference import score_dataframe, write_prediction_artifacts
from topic_alignment.modeling import create_stage_run_dir, load_model_with_adapter, save_run_metadata
from topic_alignment.prompting import build_kto_rows, build_preference_row, candidate_responses
from topic_alignment.training import train_dpo, train_kto, train_sft
from topic_alignment.utils import set_global_seed, write_json, write_jsonl


DEFAULT_CONFIG = "configs/llama31.yaml"


def _sft_paths(config: dict, seed: int, train_size: str, allocation: str):
    sft_dir = create_stage_run_dir(config, seed, train_size, allocation, "sft")
    sft_adapter_dir = sft_dir / "adapter"
    tagged_test_path = sft_dir / "tagged_test_predictions.csv"
    return sft_dir, sft_adapter_dir, tagged_test_path


def _load_tagged_test(tagged_test_path):
    return pd.read_csv(tagged_test_path).drop(
        columns=["predicted_label", "predicted_response", "is_correct", "hard_negative_label"],
        errors="ignore",
    )


def _apply_tags(dataframe: pd.DataFrame, threshold: float) -> pd.DataFrame:
    tagged = dataframe.copy()
    tagged["is_hard_primary"] = tagged["score_margin"] <= threshold
    tagged["is_hard_secondary"] = tagged["predicted_label"] != tagged["label"]
    return tagged


def _train_sft_stage(args: argparse.Namespace, config: dict) -> None:
    prepare_dataset_artifacts(config)
    run_dir = create_stage_run_dir(config, args.seed, args.train_size, args.allocation, "sft")
    save_run_metadata(
        run_dir,
        {
            "method": "sft",
            "seed": args.seed,
            "train_size": args.train_size,
            "data_allocation": args.allocation,
        },
    )

    source_train_df = load_prepared_subset(config, args.seed, args.train_size)
    allocation_payload = allocate_training_rows(source_train_df, config, seed=args.seed, data_allocation=args.allocation)
    train_df = allocation_payload["sft_train"]
    preference_source_df = allocation_payload["preference_source"]
    test_df = load_prepared_split(config, "test")
    write_json(run_dir / "data_allocation.json", allocation_payload["summary"])
    train_df.to_csv(run_dir / "sft_train_split.csv", index=False)
    preference_source_df.to_csv(run_dir / "preference_source_split.csv", index=False)

    adapter_dir = train_sft(config, train_df, test_df, run_dir, seed=args.seed)
    model, tokenizer = load_model_with_adapter(config, adapter_dir, trainable=False)
    set_global_seed(args.seed)

    predictions_train = score_dataframe(
        model,
        tokenizer,
        train_df,
        config,
        progress_label="sft-train",
    )
    predictions_preference_source = score_dataframe(
        model,
        tokenizer,
        preference_source_df,
        config,
        progress_label="sft-preference-source",
    )
    predictions_test = score_dataframe(
        model,
        tokenizer,
        test_df,
        config,
        progress_label="sft-test",
    )

    write_prediction_artifacts(predictions_train, run_dir, "train", config)
    write_prediction_artifacts(
        predictions_preference_source,
        run_dir,
        "preference_source",
        config,
    )
    write_prediction_artifacts(predictions_test, run_dir, "test", config)


def _tag_hard_cases(args: argparse.Namespace, config: dict) -> None:
    sft_dir = create_stage_run_dir(config, args.seed, args.train_size, args.allocation, "sft")
    missing = [
        path.name
        for path in [
            sft_dir / "predictions_train.csv",
            sft_dir / "predictions_test.csv",
        ]
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(
            f"Missing SFT prediction files in {sft_dir}. Run `python scripts/train.py sft` first. Missing: {missing}"
        )

    train_df = pd.read_csv(sft_dir / "predictions_train.csv")
    test_df = pd.read_csv(sft_dir / "predictions_test.csv")

    threshold = float(train_df["score_margin"].quantile(float(config["difficulty"]["primary"]["margin_quantile"])))
    write_json(
        sft_dir / "difficulty_definition.json",
        {
            "primary_method": "low_margin",
            "threshold_source": "sft train score_margin quantile",
            "threshold_quantile": config["difficulty"]["primary"]["margin_quantile"],
            "threshold_value": threshold,
            "secondary_method": "pilot_misclassified",
        },
    )

    tagged_test = _apply_tags(test_df, threshold)
    tagged_test.to_csv(sft_dir / "tagged_test_predictions.csv", index=False)


def _build_preferences(args: argparse.Namespace, config: dict) -> None:
    labels = canonical_label_ids(config)
    sft_dir = create_stage_run_dir(config, args.seed, args.train_size, args.allocation, "sft")
    preferences_dir = create_stage_run_dir(config, args.seed, args.train_size, args.allocation, "preferences")
    predictions_source_path = sft_dir / "predictions_preference_source.csv"
    if not predictions_source_path.exists():
        predictions_source_path = sft_dir / "predictions_train.csv"
    if not predictions_source_path.exists():
        raise RuntimeError(
            f"Missing {predictions_source_path}. Run `python scripts/train.py sft` first."
        )
    predictions_train = pd.read_csv(predictions_source_path)

    rng = random.Random(args.seed)
    summaries: dict[str, dict[str, int | str]] = {}
    kto_strategy = str(config.get("kto", {}).get("dataset_strategy", "all_wrong"))
    kto_rows = []
    for record in predictions_train.to_dict("records"):
        gold_label = record["label"]
        for kto_row in build_kto_rows(record["text"], gold_label, config):
            kto_rows.append(
                {
                    "example_id": record["example_id"],
                    "gold_label": gold_label,
                    "completion_label": gold_label if bool(kto_row["label"]) else kto_row["completion"],
                    "completion_kind": "desirable" if bool(kto_row["label"]) else "undesirable",
                    "strategy": kto_strategy,
                    **kto_row,
                }
            )

    kto_output_df = pd.DataFrame(kto_rows)
    if not kto_output_df.empty:
        completion_map = {response: label for label, response in candidate_responses(config).items()}
        kto_output_df["completion_label"] = kto_output_df["completion"].map(completion_map)
    kto_output_df.to_csv(preferences_dir / f"kto_{kto_strategy}.csv", index=False)
    write_jsonl(preferences_dir / f"kto_{kto_strategy}.jsonl", kto_rows)
    summaries["kto"] = {
        "strategy": kto_strategy,
        "row_count": int(len(kto_output_df)),
        "desirable_row_count": int(kto_output_df["label"].sum()) if not kto_output_df.empty else 0,
        "undesirable_row_count": int((~kto_output_df["label"]).sum()) if not kto_output_df.empty else 0,
        "data_allocation": str(args.allocation),
    }

    for strategy in config["dpo"]["strategies"]:
        rows = []
        fallback_count = 0
        for record in predictions_train.to_dict("records"):
            gold_label = record["label"]
            other_labels = [label for label in labels if label != gold_label]
            if strategy == "random_wrong":
                rejected_label = rng.choice(other_labels)
            elif strategy == "hard_negative":
                rejected_label = record.get("hard_negative_label")
                if rejected_label == gold_label or rejected_label not in other_labels:
                    fallback_count += 1
                    rejected_label = max(
                        other_labels,
                        key=lambda candidate: record[f"score__{candidate}"],
                    )
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            preference_row = {
                "example_id": record["example_id"],
                "gold_label": gold_label,
                "rejected_label": rejected_label,
                "strategy": strategy,
            }
            preference_row.update(build_preference_row(record["text"], gold_label, rejected_label, config))
            rows.append(preference_row)

        output_df = pd.DataFrame(rows)
        output_df.to_csv(preferences_dir / f"{strategy}.csv", index=False)
        write_jsonl(preferences_dir / f"{strategy}.jsonl", rows)
        summaries[strategy] = {
            "row_count": int(len(output_df)),
            "fallback_count": int(fallback_count),
            "data_allocation": str(args.allocation),
        }

    write_json(preferences_dir / "summary.json", summaries)


def _write_test_outputs(
    config: dict,
    adapter_dir,
    run_dir,
    tagged_test_path,
    seed: int,
    progress_label: str,
    method: str,
    preference_strategy: str,
    allocation: str,
) -> None:
    model, tokenizer = load_model_with_adapter(config, adapter_dir, trainable=False)
    set_global_seed(seed)
    test_df = _load_tagged_test(tagged_test_path)
    predictions_test = score_dataframe(
        model,
        tokenizer,
        test_df,
        config,
        progress_label=progress_label,
    )
    write_prediction_artifacts(predictions_test, run_dir, "test", config)
    write_json(
        run_dir / "metrics.json",
        {
            "metadata": {
                "method": method,
                "preference_strategy": preference_strategy,
                "data_allocation": allocation,
            },
            "test": compute_metrics_bundle(predictions_test, config),
        },
    )


def _train_dpo_stage(args: argparse.Namespace, config: dict) -> None:
    method = "dpo" if args.start_from == "base" else "sft_dpo"
    run_dir = create_stage_run_dir(
        config,
        args.seed,
        args.train_size,
        args.allocation,
        f"dpo_{args.preference_strategy}_from_{args.start_from}",
    )
    save_run_metadata(
        run_dir,
        {
            "method": method,
            "seed": args.seed,
            "train_size": args.train_size,
            "preference_strategy": args.preference_strategy,
            "start_from": args.start_from,
            "data_allocation": args.allocation,
        },
    )

    preferences_dir = create_stage_run_dir(config, args.seed, args.train_size, args.allocation, "preferences")
    preference_path = preferences_dir / f"{args.preference_strategy}.csv"
    if not preference_path.exists():
        raise RuntimeError(f"Missing {preference_path}. Run `python scripts/train.py build-preferences` first.")
    preference_df = pd.read_csv(preference_path)

    _, sft_adapter_dir, tagged_test_path = _sft_paths(config, args.seed, args.train_size, args.allocation)
    if args.start_from == "sft" and not sft_adapter_dir.exists():
        raise RuntimeError(f"Missing SFT adapter directory {sft_adapter_dir}. Run `python scripts/train.py sft` first.")
    if not tagged_test_path.exists():
        raise RuntimeError(f"Missing {tagged_test_path}. Run `python scripts/train.py tag-hard-cases` first.")

    adapter_dir = train_dpo(
        config,
        preference_df,
        run_dir,
        seed=args.seed,
        start_from=args.start_from,
        sft_adapter_dir=sft_adapter_dir if args.start_from == "sft" else None,
    )
    _write_test_outputs(
        config,
        adapter_dir,
        run_dir,
        tagged_test_path,
        args.seed,
        progress_label="dpo-test",
        method=method,
        preference_strategy=args.preference_strategy,
        allocation=args.allocation,
    )


def _train_kto_stage(args: argparse.Namespace, config: dict) -> None:
    dataset_strategy = str(args.dataset_strategy or config.get("kto", {}).get("dataset_strategy", "all_wrong"))
    expected_strategy = str(config.get("kto", {}).get("dataset_strategy", "all_wrong"))
    if dataset_strategy != expected_strategy:
        raise ValueError(f"Unsupported KTO dataset strategy: {dataset_strategy}. Expected {expected_strategy}.")

    method = "kto" if args.start_from == "base" else "sft_kto"
    run_dir = create_stage_run_dir(
        config,
        args.seed,
        args.train_size,
        args.allocation,
        f"kto_{dataset_strategy}_from_{args.start_from}",
    )
    save_run_metadata(
        run_dir,
        {
            "method": method,
            "seed": args.seed,
            "train_size": args.train_size,
            "preference_strategy": dataset_strategy,
            "start_from": args.start_from,
            "data_allocation": args.allocation,
        },
    )

    preferences_dir = create_stage_run_dir(config, args.seed, args.train_size, args.allocation, "preferences")
    kto_path = preferences_dir / f"kto_{dataset_strategy}.csv"
    if not kto_path.exists():
        raise RuntimeError(f"Missing {kto_path}. Run `python scripts/train.py build-preferences` first.")
    kto_df = pd.read_csv(kto_path)

    _, sft_adapter_dir, tagged_test_path = _sft_paths(config, args.seed, args.train_size, args.allocation)
    if args.start_from == "sft" and not sft_adapter_dir.exists():
        raise RuntimeError(f"Missing SFT adapter directory {sft_adapter_dir}. Run `python scripts/train.py sft` first.")
    if not tagged_test_path.exists():
        raise RuntimeError(f"Missing {tagged_test_path}. Run `python scripts/train.py tag-hard-cases` first.")

    adapter_dir = train_kto(
        config,
        kto_df,
        run_dir,
        seed=args.seed,
        start_from=args.start_from,
        sft_adapter_dir=sft_adapter_dir if args.start_from == "sft" else None,
    )
    _write_test_outputs(
        config,
        adapter_dir,
        run_dir,
        tagged_test_path,
        args.seed,
        progress_label="kto-test",
        method=method,
        preference_strategy=dataset_strategy,
        allocation=args.allocation,
    )


def _run(script_name: str, *script_args: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name
    command = [sys.executable, str(script_path), *script_args]
    subprocess.run(command, check=True)


def _run_full(args: argparse.Namespace, config: dict) -> None:
    manifest = prepare_dataset_artifacts(config)
    available_sizes = manifest["available_train_sizes"]
    strategies = ["hard_negative"]
    if args.include_random_wrong:
        strategies.append("random_wrong")

    _run("prepare_data.py", "--config", args.config)
    _run("score.py", "--config", args.config)

    for seed in config["task"]["subset_seeds"]:
        for train_size in available_sizes:
            for allocation in data_allocation_variants(config):
                _run(
                    "train.py",
                    "sft",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--train-size",
                    str(train_size),
                    "--allocation",
                    allocation,
                )
                _run(
                    "train.py",
                    "tag-hard-cases",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--train-size",
                    str(train_size),
                    "--allocation",
                    allocation,
                )
                _run(
                    "train.py",
                    "build-preferences",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--train-size",
                    str(train_size),
                    "--allocation",
                    allocation,
                )
                for strategy in strategies:
                    if allocation == "shared":
                        _run(
                            "train.py",
                            "dpo",
                            "--config",
                            args.config,
                            "--seed",
                            str(seed),
                            "--train-size",
                            str(train_size),
                            "--preference-strategy",
                            strategy,
                            "--start-from",
                            "base",
                            "--allocation",
                            allocation,
                        )
                    _run(
                        "train.py",
                        "dpo",
                        "--config",
                        args.config,
                        "--seed",
                        str(seed),
                        "--train-size",
                        str(train_size),
                        "--preference-strategy",
                        strategy,
                        "--start-from",
                        "sft",
                        "--allocation",
                        allocation,
                    )
                if allocation == "shared":
                    _run(
                        "train.py",
                        "kto",
                        "--config",
                        args.config,
                        "--seed",
                        str(seed),
                        "--train-size",
                        str(train_size),
                        "--start-from",
                        "base",
                        "--allocation",
                        allocation,
                    )
                _run(
                    "train.py",
                    "kto",
                    "--config",
                    args.config,
                    "--seed",
                    str(seed),
                    "--train-size",
                    str(train_size),
                    "--start-from",
                    "sft",
                    "--allocation",
                    allocation,
                )

    _run("aggregate.py", "--config", args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and orchestrate SFT, DPO, and KTO experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sft_parser = subparsers.add_parser("sft", description="Train one SFT run.")
    sft_parser.add_argument("--config", default=DEFAULT_CONFIG)
    sft_parser.add_argument("--seed", type=int, required=True)
    sft_parser.add_argument("--train-size", required=True)
    sft_parser.add_argument("--allocation", default="shared")

    tag_parser = subparsers.add_parser("tag-hard-cases", description="Tag hard test cases from SFT predictions.")
    tag_parser.add_argument("--config", default=DEFAULT_CONFIG)
    tag_parser.add_argument("--seed", type=int, required=True)
    tag_parser.add_argument("--train-size", required=True)
    tag_parser.add_argument("--allocation", default="shared")

    preference_parser = subparsers.add_parser("build-preferences", description="Build DPO/KTO datasets from SFT predictions.")
    preference_parser.add_argument("--config", default=DEFAULT_CONFIG)
    preference_parser.add_argument("--seed", type=int, required=True)
    preference_parser.add_argument("--train-size", required=True)
    preference_parser.add_argument("--allocation", default="shared")

    dpo_parser = subparsers.add_parser("dpo", description="Train DPO or SFT->DPO.")
    dpo_parser.add_argument("--config", default=DEFAULT_CONFIG)
    dpo_parser.add_argument("--seed", type=int, required=True)
    dpo_parser.add_argument("--train-size", required=True)
    dpo_parser.add_argument("--preference-strategy", required=True, choices=["random_wrong", "hard_negative"])
    dpo_parser.add_argument("--start-from", required=True, choices=["base", "sft"])
    dpo_parser.add_argument("--allocation", default="shared")

    kto_parser = subparsers.add_parser("kto", description="Train KTO or SFT->KTO.")
    kto_parser.add_argument("--config", default=DEFAULT_CONFIG)
    kto_parser.add_argument("--seed", type=int, required=True)
    kto_parser.add_argument("--train-size", required=True)
    kto_parser.add_argument("--dataset-strategy", default=None)
    kto_parser.add_argument("--start-from", required=True, choices=["base", "sft"])
    kto_parser.add_argument("--allocation", default="shared")

    full_parser = subparsers.add_parser("full", description="Run the full configured experiment grid.")
    full_parser.add_argument("--config", default=DEFAULT_CONFIG)
    full_parser.add_argument("--include-random-wrong", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "sft":
        _train_sft_stage(args, config)
    elif args.command == "tag-hard-cases":
        _tag_hard_cases(args, config)
    elif args.command == "build-preferences":
        _build_preferences(args, config)
    elif args.command == "dpo":
        _train_dpo_stage(args, config)
    elif args.command == "kto":
        _train_kto_stage(args, config)
    elif args.command == "full":
        _run_full(args, config)
    else:
        raise ValueError(f"Unsupported train command: {args.command}")


if __name__ == "__main__":
    main()
