import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work3.continual_rlvr import admit_method
from work3.continual_rlvr_data import PROMPT_BOUND, normalize, prompt_audit, score, sha256_file, split_rows
from work3.continual_rlvr_protocol import ARMS, SEEDS, method_verdict, necessity_verdict
from work3.wiki_rsi_progress import Progress, board_runs


def b_row(question, phrase):
    return {
        "messages": [{"role": "user", "content": f"{question}\nFinish with {phrase}."}],
        "ground_truth": json.dumps({"func_name": "validate_end", "end_phrase": phrase}),
        "constraint": "Finish with {end phrase}.",
    }


class MeasuredTokenizer:
    def __init__(self, lengths=None):
        self.lengths = lengths or {}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [1] * self.lengths.get(messages[0]["content"], 10)


class ContinualDataTests(unittest.TestCase):
    def test_constraint_variants_stay_together(self):
        rows = [b_row("Question 1", "done"), b_row("Question 1", "complete"), b_row("Question 2", "done")]
        with patch("work3.continual_rlvr_data.SIZES", {"b_train": 1, "b_dev": 1}):
            result, maxima, replacements = split_rows(rows, "b", MeasuredTokenizer())
            self.assertEqual((result, maxima, replacements), split_rows(reversed(rows), "b", MeasuredTokenizer()))
        self.assertNotEqual(result["b_train"][0]["group"], result["b_dev"][0]["group"])
        self.assertTrue(score(normalize("b", rows[0]), "The explanation is done"))
        self.assertFalse(score(normalize("b", rows[0]), "The explanation is undone"))

    def test_exact_math_reward_does_not_accept_vectors_or_first_of_two_boxes(self):
        case = {"task": "a", "target": "1/2"}
        self.assertTrue(score(case, r"The exact answer is \boxed{\frac{-1}{-2}}."))
        self.assertFalse(score(case, r"\boxed{0.5} but the answer is \boxed{1,2}"))
        self.assertFalse(score({"task": "a", "target": "12"}, r"\boxed{1,2}"))

    def test_token_overflow_replaces_only_training_from_unused_groups(self):
        rows = [b_row(f"Question {i}", "done") for i in range(6)]
        sizes = {"b_anchor": 1, "b_train": 2, "b_dev": 1}
        with patch("work3.continual_rlvr_data.SIZES", sizes):
            original, _, _ = split_rows(rows, "b", MeasuredTokenizer())
            old = original["b_train"][0]
            tokenizer = MeasuredTokenizer({old["prompt"]: PROMPT_BOUND + 1})
            updated, maxima, replacements = split_rows(rows, "b", tokenizer)
            self.assertEqual(original["b_dev"], updated["b_dev"])
            self.assertEqual(original["b_anchor"], updated["b_anchor"])
            self.assertEqual(original["b_train"][1], updated["b_train"][1])
            self.assertEqual(maxima, prompt_audit(updated, tokenizer))
            self.assertEqual(replacements[0]["removed_id"], old["id"])
            self.assertEqual(replacements[0]["removed_tokens"], PROMPT_BOUND + 1)
            self.assertEqual(replacements[0]["replacement_id"], updated["b_train"][0]["id"])
            old_groups = {case["group"] for cases in original.values() for case in cases}
            self.assertNotIn(updated["b_train"][0]["group"], old_groups)
            with self.assertRaisesRegex(ValueError, "frozen holdout cannot change"):
                split_rows(rows, "b", MeasuredTokenizer({original["b_dev"][0]["prompt"]: PROMPT_BOUND + 1}))


class ContinualGateTests(unittest.TestCase):
    def test_necessity_and_c2_admission(self):
        before = {"a": {"a": [False] * 4},
                  "b": {str(i): [True] * 32 for i in range(20)}}
        after = {"a": {"a": [True] * 4},
                 "b": {str(i): [i != 0] * 32 for i in range(20)}}
        self.assertEqual(necessity_verdict(before, after)["verdict"], "GO")
        splits = {"a_dev": [{"id": "a"}], "b_dev": [{"id": key} for key in before["b"]]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            raw = path / "necessity_rollouts.json"
            raw.write_text(json.dumps({"before": before, "after": after}), encoding="utf-8")
            summary = path / "wiki_rsi_continual_necessity_summary.json"
            summary.write_text(json.dumps({"provenance": {}, "verdict": "GO",
                                           "rollouts_sha256": sha256_file(raw)}), encoding="utf-8")
            admit_method(path, {}, splits)
            with self.assertRaisesRegex(ValueError, "does not match"):
                admit_method(path, {"seed": 1}, splits)
            raw.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                admit_method(path, {}, splits)
        after["b"]["0"][16:] = [True] * 16
        self.assertEqual(necessity_verdict(before, after)["verdict"], "NO_GO")

    def test_method_requires_learning_after_its_own_start(self):
        cost = {key: 100 for key in ("a_rollout_tokens", "a_update_tokens", "b_rollout_tokens",
                                     "b_update_tokens", "train_seconds")}
        records = {seed: {arm: {"b_pass1": 0.5 if arm == "guard" else 0.4, "b_curve": {0: 0.3},
                                "a_after_a_pass1": 0.5, "b_final": {"b": [arm == "guard"]},
                                "b_after_a_pass4": 0.1 * index, "calibration_ratio": 0.1 * index,
                                "cost": cost}
                         for index, arm in enumerate(ARMS)} for seed in SEEDS}
        self.assertEqual(method_verdict(records)["verdict"], "GO")
        records[SEEDS[0]]["guard"]["b_curve"][0] = 0.6
        self.assertEqual(method_verdict(records)["verdict"], "NO_GO")

    def test_board_sees_continual_phase(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "continual-c1"
            path.mkdir()
            progress = Progress(path, "continual_necessity")
            progress.update("running", 10, 48, detail="continual_train_a", metrics={"arm_index": 2})
            runs = board_runs(Path(root), progress.data["updated_at"])
        self.assertEqual(runs[0]["phases"]["continual_necessity"]["metrics"], {"arm_index": 2})
