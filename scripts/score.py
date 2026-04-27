from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from topic_alignment.config import load_config
from topic_alignment.data import load_prepared_split, prepare_dataset_artifacts
from topic_alignment.evaluation import compute_metrics_bundle
from topic_alignment.inference import score_dataframe, write_prediction_artifacts
from topic_alignment.modeling import create_run_dir, load_base_model, prepare_for_inference, save_run_metadata
from topic_alignment.utils import set_global_seed, write_json


DEFAULT_CONFIG = "configs/llama31.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained baseline inference.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_dataset_artifacts(config)
    run_dir = create_run_dir(config, "baseline")
    save_run_metadata(
        run_dir,
        {
            "method": "pretrained_baseline",
            "model": config["model"]["base_model_id"],
            "config_hash": config.get("_config_hash"),
        },
    )

    model, tokenizer = load_base_model(config)
    model = prepare_for_inference(model)
    set_global_seed(int(config["task"]["split_seed"]))

    test_df = load_prepared_split(config, "test")
    predictions_test = score_dataframe(
        model,
        tokenizer,
        test_df,
        config,
        progress_label="baseline-test",
    )

    write_prediction_artifacts(predictions_test, run_dir, "test", config)
    write_json(
        run_dir / "metrics.json",
        {
            "metadata": {"method": "pretrained_baseline"},
            "test": compute_metrics_bundle(predictions_test, config),
        },
    )


if __name__ == "__main__":
    main()
