from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from topic_alignment.config import load_config
from topic_alignment.data import prepare_dataset_artifacts


DEFAULT_CONFIG = "configs/llama31.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the configured classification dataset and train-size subsets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = prepare_dataset_artifacts(config)
    print(manifest)


if __name__ == "__main__":
    main()
