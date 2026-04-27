from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from topic_alignment.config import load_config
from topic_alignment.data import make_splits_and_subsets, prepare_dataset_artifacts
from topic_alignment.utils import resolve_path


class ConfigDataTests(unittest.TestCase):
    def test_canonical_configs_use_raw_or_sample_inputs(self) -> None:
        for config_path in ["configs/llama31.yaml", "configs/gemma3.yaml"]:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertEqual(config["task"]["samples_dir"], "data/raw_or_samples")
                self.assertTrue(resolve_path(config["task"]["samples_dir"]).exists())

    def test_provided_subset_mode_creates_train_and_test_only(self) -> None:
        config = load_config("configs/llama31.yaml")
        split_map, subsets = make_splits_and_subsets(config)

        self.assertEqual(set(split_map), {"train", "test"})
        self.assertEqual(len(split_map["train"]), 500)
        self.assertGreater(len(split_map["test"]), 0)
        self.assertEqual(set(subsets), {50, 100, 250, 500, "full"})

    def test_generated_manifest_uses_repo_relative_paths(self) -> None:
        config = load_config("configs/llama31.yaml")
        config["project"]["data_root"] = "data/processed_test"
        generated_root = resolve_path(config["project"]["data_root"])

        try:
            manifest = prepare_dataset_artifacts(config)
            self.assertFalse(Path(manifest["processed_dir"]).is_absolute())
            self.assertEqual(manifest["splits_dir"], "data/processed_test/superframe_classification_ds3_task6/splits")
            self.assertEqual(
                sorted(path.name for path in (generated_root / config["task"]["name"] / "splits").glob("*.csv")),
                ["test.csv", "train.csv"],
            )
        finally:
            shutil.rmtree(generated_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
