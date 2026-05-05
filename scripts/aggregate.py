from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from topic_alignment.config import load_config
from topic_alignment.data import prepare_dataset_artifacts


DEFAULT_CONFIG = "configs/llama31.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate predictions into tables and plots.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--task", default=None, help="Task key from the config task_registry.")
    args = parser.parse_args()

    config = load_config(args.config, task_key=args.task)
    prepare_dataset_artifacts(config)

    from topic_alignment.reporting import (
        build_results_table,
        plot_confusion_matrices,
        plot_training_curves,
        write_per_class_breakdown,
        write_results_tables,
    )

    results_df = build_results_table(config)
    table_paths = write_results_tables(config, results_df)
    table_paths["per_class_table"] = write_per_class_breakdown(config)
    figure_paths = (
        plot_training_curves(config, results_df)
        + plot_confusion_matrices(config)
    )
    print({"tables": table_paths, "figures": figure_paths})


if __name__ == "__main__":
    main()
