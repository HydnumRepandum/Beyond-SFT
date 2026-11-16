from __future__ import annotations

import json
import random
import shutil
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from topic_alignment.config import learning_rate_tag, load_config, post_learning_rates, stage_budget
from topic_alignment.data import load_prepared_subset, prepare_dataset_artifacts
from topic_alignment.generation import INVALID, majority_vote, parse_lenient, parse_strict
from topic_alignment.preferences import oof_fold_assignments, rejected_label_agreement, select_rejected_labels
from topic_alignment.stats import (
    _macro_f1,
    collect_runs,
    contrast_table,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
    summary_table,
)
from topic_alignment.training import stage_overrides
from topic_alignment.utils import resolve_path, write_json

LABELS = ["A", "B", "C", "D"]


class ProtocolConfigTests(unittest.TestCase):
    def test_revision_config_extends_base_and_keeps_outputs_separate(self) -> None:
        for name in ["llama31", "gemma3"]:
            base = load_config(f"configs/{name}.yaml")
            revision = load_config(f"configs/{name}_revision.yaml")
            self.assertEqual(revision["model"], base["model"])
            self.assertEqual(revision["sft"], base["sft"])
            self.assertNotEqual(revision["project"]["outputs_root"], base["project"]["outputs_root"])
            self.assertNotEqual(revision["project"]["data_root"], base["project"]["data_root"])
            self.assertEqual(revision["task"]["subset_seeds"], [11, 22, 33, 44, 55])
            self.assertEqual(base["task"]["subset_seeds"], [11, 22, 33])

    def test_legacy_config_has_no_budget_or_learning_rate_grid(self) -> None:
        base = load_config("configs/llama31.yaml")
        self.assertIsNone(stage_budget(base, "sft", 100))
        self.assertEqual(post_learning_rates(base), [None])
        self.assertEqual(learning_rate_tag(None), "")

    def test_matched_budget_is_identical_for_all_post_sft_stages(self) -> None:
        config = load_config("configs/llama31_revision.yaml")
        post = stage_budget(config, "sft", 100)
        # 100 examples x 2 epochs / (4 x 4) = 12.5 -> 13 steps
        self.assertEqual(post, {"max_steps": 13, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4})
        # From the base model the budget equals the SFT stage: 100 x 3 / (4 x 2) = 37.5 -> 38
        self.assertEqual(stage_budget(config, "base", 100)["max_steps"], 38)

    def test_stage_overrides_apply_budget_and_learning_rate(self) -> None:
        config = load_config("configs/llama31_revision.yaml")
        resolved = stage_overrides(config["dpo"], 2e-5, stage_budget(config, "sft", 50))
        self.assertEqual(resolved["max_steps"], 7)
        self.assertEqual(resolved["learning_rate"], 2e-5)
        self.assertEqual(stage_overrides(config["dpo"], None, None)["max_steps"], -1)
        self.assertEqual(learning_rate_tag(2e-5), "__lr2e-5")


class ResamplingTests(unittest.TestCase):
    def test_each_seed_draws_a_different_stratified_subset(self) -> None:
        config = load_config("configs/llama31_revision.yaml")
        config["project"]["data_root"] = "data/processed_test_revision"
        root = resolve_path(config["project"]["data_root"])
        try:
            prepare_dataset_artifacts(config)
            first = load_prepared_subset(config, 11, 100)
            second = load_prepared_subset(config, 22, 100)
            self.assertEqual(len(first), 100)
            self.assertEqual(len(second), 100)
            self.assertNotEqual(set(first["example_id"]), set(second["example_id"]))
            self.assertEqual(set(first["label"]), set(LABELS))
            # The largest size is the whole pool and cannot differ between seeds.
            self.assertEqual(
                set(load_prepared_subset(config, 11, 500)["example_id"]),
                set(load_prepared_subset(config, 22, 500)["example_id"]),
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class PreferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.in_sample = {"score__A": -0.1, "score__B": -3.0, "score__C": -2.0, "score__D": -5.0}
        self.oof = {"score__A": -1.5, "score__B": -0.4, "score__C": -2.0, "score__D": -5.0}

    def _select(self, strategy: str) -> list[str]:
        return select_rejected_labels(
            strategy, "A", LABELS, random.Random(0), in_sample_record=self.in_sample, oof_record=self.oof
        )

    def test_strategies_pick_expected_rejected_labels(self) -> None:
        self.assertEqual(self._select("hard_negative"), ["C"])
        self.assertEqual(self._select("hard_negative_oof"), ["B"])
        self.assertEqual(self._select("all_wrong"), ["B", "C", "D"])
        self.assertIn(self._select("random_wrong")[0], ["B", "C", "D"])

    def test_oof_strategy_requires_oof_scores(self) -> None:
        with self.assertRaises(ValueError):
            select_rejected_labels("hard_negative_oof", "A", LABELS, random.Random(0), in_sample_record=self.in_sample)

    def test_folds_are_stratified_and_shrink_to_rarest_class(self) -> None:
        frame = pd.DataFrame({"label": ["A"] * 20 + ["B"] * 20 + ["C"] * 3})
        folds = oof_fold_assignments(frame, folds=5, seed=1)
        self.assertEqual(sorted(folds.unique()), [0, 1, 2])
        self.assertEqual(sorted(folds[frame["label"] == "C"]), [0, 1, 2])
        with self.assertRaises(ValueError):
            oof_fold_assignments(pd.DataFrame({"label": ["A", "A", "B"]}), folds=5, seed=1)

    def test_rejected_label_agreement(self) -> None:
        hard = pd.DataFrame({"example_id": [1, 2, 3, 4], "rejected_label": ["B", "B", "C", "D"]})
        other = pd.DataFrame({"example_id": [1, 2, 3, 4], "rejected_label": ["B", "C", "C", "B"]})
        every = pd.DataFrame({"example_id": [1, 1, 2, 2], "rejected_label": ["B", "C", "B", "C"]})
        agreement = rejected_label_agreement(
            {"hard_negative": hard, "random_wrong": other, "all_wrong": every}, "hard_negative"
        )
        self.assertEqual(agreement, {"random_wrong": 0.5})


class GenerationParsingTests(unittest.TestCase):
    def test_strict_and_lenient_parsing(self) -> None:
        self.assertEqual(parse_strict(' {"label":"B"}\n', LABELS), "B")
        self.assertEqual(parse_strict('{"label": "B"}', LABELS), INVALID)
        self.assertEqual(parse_lenient('{"label": "B"}', LABELS), "B")
        self.assertEqual(parse_lenient("C: ECONOMICS", LABELS), "C")
        self.assertEqual(parse_lenient("The answer is unclear", LABELS), INVALID)
        self.assertEqual(parse_lenient("Answer", LABELS), INVALID)

    def test_majority_vote(self) -> None:
        self.assertEqual(majority_vote(["A", "B", "B"]), "B")
        self.assertEqual(majority_vote([INVALID, "C", INVALID]), "C")
        self.assertEqual(majority_vote(["B", "A"]), "B")
        self.assertEqual(majority_vote([INVALID]), INVALID)


class StatisticsTests(unittest.TestCase):
    def test_macro_f1_matches_sklearn_including_invalid_predictions(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 4, size=300)
        y_pred = rng.integers(0, 5, size=300)  # 4 encodes an invalid prediction
        expected = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        self.assertAlmostEqual(float(_macro_f1(y_true, y_pred, 4)), expected, places=12)

    def test_mcnemar_and_holm(self) -> None:
        a = np.array([True] * 10 + [False] * 2)
        b = np.array([False] * 10 + [True] * 2)
        result = mcnemar_exact(a, b)
        self.assertEqual((result["only_a_correct"], result["only_b_correct"]), (10, 2))
        self.assertAlmostEqual(result["p_value"], 0.03857421875)
        self.assertEqual(mcnemar_exact(a, a)["p_value"], 1.0)
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def _pairs(self, treatment_accuracy: float, seeds: int = 5, items: int = 400):
        rng = np.random.default_rng(1)
        pairs = []
        for _ in range(seeds):
            y_true = rng.integers(0, 4, size=items)
            reference = np.where(rng.random(items) < 0.6, y_true, (y_true + 1) % 4)
            treatment = np.where(rng.random(items) < treatment_accuracy, y_true, (y_true + 1) % 4)
            pairs.append((y_true, treatment, reference))
        return pairs

    def test_bootstrap_detects_a_real_gain_and_not_a_null_one(self) -> None:
        gain = paired_bootstrap(self._pairs(0.75), 4, n_boot=500)
        self.assertGreater(gain["accuracy"]["ci_low"], 0.0)
        self.assertGreater(gain["macro_f1"]["ci_low"], 0.0)
        null = paired_bootstrap(self._pairs(0.6), 4, n_boot=500)
        self.assertLess(null["accuracy"]["ci_low"], 0.0)
        self.assertGreater(null["accuracy"]["ci_high"], 0.0)

    def test_contrast_table_pairs_runs_by_seed_and_learning_rate(self) -> None:
        config = load_config("configs/llama31_revision.yaml")
        config["project"]["outputs_root"] = "outputs/_test_stats"
        root = resolve_path(config["project"]["outputs_root"])
        rng = np.random.default_rng(2)
        gold = rng.choice(LABELS, size=200)

        def write_run(seed: int, stage: str, metadata: dict, accuracy: float) -> None:
            run_dir = root / "runs" / f"seed_{seed}" / "train_100" / "allocation_shared" / stage
            run_dir.mkdir(parents=True)
            predicted = np.where(rng.random(200) < accuracy, gold, "D")
            pd.DataFrame({"example_id": range(200), "label": gold, "predicted_label": predicted}).to_csv(
                run_dir / "predictions_test.csv", index=False
            )
            write_json(
                run_dir / "metadata.json",
                {"seed": seed, "train_size": "100", "data_allocation": "shared", **metadata},
            )

        try:
            for seed in [11, 22, 33]:
                write_run(seed, "sft", {"method": "sft"}, 0.5)
                for rate in [5e-6, 1e-4]:
                    tag = learning_rate_tag(rate)
                    shared = {"learning_rate_override": rate}
                    write_run(seed, f"sft_continued_from_sft{tag}", {"method": "sft_sft", **shared}, 0.5)
                    write_run(
                        seed,
                        f"dpo_hard_negative_from_sft{tag}",
                        {"method": "sft_dpo", "preference_strategy": "hard_negative", **shared},
                        0.9 if rate == 1e-4 else 0.5,
                    )
            runs = collect_runs(config)
            self.assertEqual(len(runs), 15)
            self.assertEqual(int(summary_table(config, runs)["seed_count"].min()), 3)

            matched = contrast_table(config, runs, "sft_sft", n_boot=200)
            self.assertEqual(len(matched), 2)
            self.assertTrue((matched["seed_count"] == 3).all())
            by_rate = matched.set_index("learning_rate")
            self.assertGreater(by_rate.loc["0.0001", "accuracy_ci_low"], 0.2)
            self.assertLess(by_rate.loc["5e-06", "accuracy_ci_low"], 0.0)
            self.assertIn("accuracy_p_holm", matched.columns)

            versus_sft = contrast_table(config, runs, "sft", n_boot=200)
            self.assertEqual(set(versus_sft["method"]), {"sft_sft", "sft_dpo"})
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
