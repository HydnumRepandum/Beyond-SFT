from __future__ import annotations

import inspect
import shutil
import unittest
from pathlib import Path

from topic_alignment.config import (
    available_task_keys,
    canonical_label_ids,
    load_config,
    raw_to_canonical_map,
    task_instruction_text,
    task_metadata,
)
from topic_alignment.data import make_splits_and_subsets, prepare_dataset_artifacts
from topic_alignment.training import train_sft
from topic_alignment.utils import resolve_path


class ConfigDataTests(unittest.TestCase):
    external_task_root = Path("/Users/ematheve/Desktop/dataset task RL")

    def test_canonical_configs_use_raw_or_sample_inputs(self) -> None:
        for config_path in ["configs/llama31.yaml", "configs/gemma3.yaml"]:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertTrue(str(config["task"]["samples_dir"]).startswith("data/raw_or_samples"))
                self.assertTrue(resolve_path(config["task"]["samples_dir"]).exists())

    def test_canonical_configs_do_not_use_test_split_for_trainer_eval(self) -> None:
        for config_path in ["configs/llama31.yaml", "configs/gemma3.yaml"]:
            with self.subTest(config_path=config_path):
                config = load_config(config_path)
                self.assertEqual(str(config["sft"]["evaluation_strategy"]), "no")
                self.assertNotIn("per_device_eval_batch_size", config["sft"])
                self.assertEqual(str(config["dpo"]["evaluation_strategy"]), "no")
                self.assertEqual(str(config["kto"]["evaluation_strategy"]), "no")

    def test_sft_training_api_does_not_accept_test_dataframe(self) -> None:
        signature = inspect.signature(train_sft)
        self.assertNotIn("test_df", signature.parameters)

    def test_task_registry_exposes_all_supported_tasks(self) -> None:
        config = load_config("configs/llama31.yaml")
        self.assertEqual(available_task_keys(config), ["ds1_task6", "ds3_task6", "ds4_task2"])

    def test_kto_desirable_weight_tracks_selected_task(self) -> None:
        expected_weights = {
            "ds3_task6": 3.0,
            "ds1_task6": 3.0,
            "ds4_task2": 2.0,
        }
        for config_path in ["configs/llama31.yaml", "configs/gemma3.yaml"]:
            for task_key, expected_weight in expected_weights.items():
                with self.subTest(config_path=config_path, task_key=task_key):
                    config = load_config(config_path, task_key=task_key)
                    self.assertEqual(float(config["kto"]["desirable_weight"]), expected_weight)
                    self.assertEqual(float(config["kto"]["undesirable_weight"]), 1.0)

    def test_ds4_task_uses_local_raw_or_sample_folder(self) -> None:
        config = load_config("configs/llama31.yaml", task_key="ds4_task2")
        self.assertEqual(config["task"]["samples_dir"], "data/raw_or_samples/ds_4__task_2")
        self.assertEqual(config["task"]["mapping_path"], "data/raw_or_samples/ds_4__task_2/dataset_task_mappings.csv")
        self.assertEqual(config["task"]["mapping_delimiter"], ";")
        self.assertEqual(config["task"]["mapping_encoding"], "cp1252")
        self.assertEqual(canonical_label_ids(config), ["A", "B", "C"])
        self.assertIn("few_shot_chain_of_thought_prompt", task_metadata(config)["prompts"])
        self.assertEqual(raw_to_canonical_map(config)["neither"], "C")

    @unittest.skipUnless(external_task_root.exists(), "external task folder is not available")
    def test_external_tasks_load_prompts_and_labels_from_mapping(self) -> None:
        ds1_config = load_config("configs/llama31.yaml", task_key="ds1_task6")
        self.assertEqual(canonical_label_ids(ds1_config), ["A", "B", "C", "D"])
        self.assertIn("few_shot_prompt", task_metadata(ds1_config)["prompts"])
        self.assertGreater(len(task_instruction_text(ds1_config) or ""), 1000)

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
