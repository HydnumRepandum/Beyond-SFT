from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from topic_alignment.config import load_config
from topic_alignment.stats import write_statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed-level summary and paired bootstrap contrasts against SFT and continued SFT."
    )
    parser.add_argument("--config", default="configs/llama31_revision.yaml")
    parser.add_argument("--task", default=None, help="Task key from the config task_registry.")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    config = load_config(args.config, task_key=args.task)
    print(json.dumps(write_statistics(config, n_boot=args.n_boot), indent=2))


if __name__ == "__main__":
    main()
