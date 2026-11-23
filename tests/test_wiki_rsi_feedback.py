"""Regression tests for automatic gates, branch reuse and live board snapshots."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from work3 import wiki_rsi_feedback as runner
from work3 import wiki_rsi_feedback_data as data
from work3 import wiki_rsi_feedback_protocol as p
from work3 import wiki_rsi_feedback_train as train
from work3.wiki_rsi_progress import Progress, atomic_json, board_runs


def scores(n=100, correct=10):
    return {f"{ds}-{i}": {"correct": int(i < correct), "valid": True, "dataset": ds, "cluster": f"{ds}-{i}"}
            for ds in p.DATASETS for i in range(n)}


class ProtocolTests(unittest.TestCase):
    def test_probe_requires_separation_and_prefers_better_branch(self):
        self.assertEqual(p.probe_choice(scores(correct=22), scores(correct=10))["choice"], "search")
        self.assertEqual(p.probe_choice(scores(correct=10), scores(correct=22))["choice"], "read")
        self.assertEqual(p.probe_choice(scores(correct=11), scores(correct=10))["choice"], "mix")

    def test_pair_minimum_and_family_diversity(self):
        self.assertFalse(p.enough_pairs({"person": 640}))
        self.assertFalse(p.enough_pairs({"person": 316, "place": 159, "work": 160}))
        self.assertTrue(p.enough_pairs({"person": 320, "place": 160, "work": 160}))

    def test_mechanism_requires_reproducible_state_interaction(self):
        reports = {(seed, state, arm): {"scores": scores(correct=(16 if (state == "early") == (arm == "search") else 10))}
                   for seed in p.SEEDS for state in ("early", "late") for arm in p.SIGNALS}
        self.assertTrue(p.mechanism_gate(reports)["ready"])
        reports[p.SEEDS[1], "late", "search"]["scores"] = scores(correct=20)
        self.assertFalse(p.mechanism_gate(reports)["ready"])

    def test_method_gate_compares_strongest_static_and_pays_probes(self):
        reports = {(seed, arm): {"scores": scores(correct=20 if arm == "adaptive" else 14 if arm == "fixed100" else 12),
                                 "training_seconds": 15, "choices": ["search", "read"]}
                   for seed in p.SEEDS for arm in p.ARMS}
        reports[p.SEEDS[0], "base"] = {"scores": scores(correct=11)}
        pilot = p.end_to_end(reports)
        self.assertEqual(pilot["chosen_fixed"], "fixed100")
        self.assertTrue(pilot["ready"])
        self.assertTrue(p.end_to_end(reports, "fixed100", final=True)["ready"])
        reports[p.SEEDS[0], "adaptive"]["training_seconds"] = 16
        self.assertFalse(p.end_to_end(reports)["ready"])
        reports[p.SEEDS[0], "adaptive"]["training_seconds"] = 15
        for seed in p.SEEDS:
            reports[seed, "adaptive"]["choices"] = ["search", "search"]
        self.assertFalse(p.end_to_end(reports)["ready"])

    def test_no_go_and_protocol_change_both_block_next_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for verdict, stored in (("NO-GO", {"v": "a"}), ("GO", {"v": "b"})):
                atomic_json(runner.receipt_path(folder, "lock"),
                            {"verdict": verdict, "contract": stored, "protocol": p.PROTOCOL})
                with self.assertRaises(ValueError):
                    runner.receipt(folder, "lock", {"v": "a"})


class DataTests(unittest.TestCase):
    def test_support_pages_stay_in_one_split_and_same_split_clusters_join(self):
        pool = {ds: [{"qid": f"{ds}-{i}", "pages": [f"{ds}-a{i}", f"{ds}-b{i}"]}
                     for i in range(6)] for ds in p.DATASETS}
        pool["hotpotqa"].extend([{"qid": "shared", "pages": ["hotpotqa-a1", "new"]},
                                 {"qid": "excluded", "pages": ["old", "also old"]}])
        with patch.dict(p.SPLITS, {"schedule": 2, "development": 2, "confirmation": 2}):
            splits = data.choose_splits(pool, {"old"})
        claimed = {}
        for split, rows in splits.items():
            for row in rows:
                for page in row["pages"]:
                    self.assertNotEqual(page, "old")
                    self.assertEqual(claimed.setdefault(page, split), split)
        clustered = data.cluster_cases([{"pages": ["a", "b"]}, {"pages": ["b", "d"]}])
        self.assertEqual(clustered[0]["cluster"], clustered[1]["cluster"])

    def test_pair_needs_both_actual_witnesses_and_hides_bridge(self):
        proposal = {"question": "Where was the director of Sample Film born?", "answer": "Sampletown",
                    "first_quote": "Sample Film was directed by Example Person.",
                    "second_quote": "Example Person was born in Sampletown."}
        first = [("Sample Film", proposal["first_quote"])]
        second = [("Example Person", proposal["second_quote"])]
        pair = data.checked_pair(proposal, "Sample Film", "Example Person", first, second, "person")
        self.assertIsNotNone(pair)
        self.assertIsNone(data.checked_pair(proposal, "Sample Film", "Example Person", first, [], "person"))
        proposal["question"] = "Where was Example Person, director of Sample Film, born?"
        self.assertIsNone(data.checked_pair(proposal, "Sample Film", "Example Person", first, second, "person"))


class RoutingTests(unittest.TestCase):
    def test_cpu_update_restores_weights_and_optimizer_moments(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                             num_hidden_layers=1, num_attention_heads=2,
                                             num_key_value_heads=2, max_position_embeddings=32))
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        trainer = train.Training(tokenizer, model, torch.device("cpu"), 4, lambda *args, **kwargs: None)
        rows = [{"examples": {"search": ([2, 4], [6]), "read": ([5, 8], [9, 10])}, "width": 4}]
        parent = trainer.snapshot()
        trainer.update(rows, [[0]], lambda *_: "search", "test")
        chosen = trainer.snapshot()
        trainer.restore(parent)
        self.assertFalse(trainer.optimizer.state)
        trainer.update(rows, [[0]], lambda *_: "read", "test")
        trainer.restore(chosen)
        for current, expected in zip(trainer.params, chosen[0], strict=True):
            self.assertTrue(torch.equal(current, expected))
        stored = trainer.optimizer.state_dict()["state"]
        for key, state in chosen[1]["state"].items():
            self.assertTrue(torch.equal(stored[key]["exp_avg"], state["exp_avg"]))
        self.assertEqual((trainer.tokens, trainer.steps), (8, 2))

    def test_probe_round_resumes_winner_from_original_checkpoint(self):
        class Trainer:
            state = "parent"
            calls = []

            def snapshot(self):
                return self.state

            def restore(self, state):
                self.state = state

            def update(self, examples, groups, route, detail):
                self.calls.append((groups, self.state, route(0, {})))
                self.state += ":" + route(0, {})

            def assess(self, cases, retriever, detail, probe):
                self.assert_probe = probe
                return scores(correct=20 if self.state.endswith("read") else 10)

        trainer = Trainer()
        decision = train.probe_round(trainer, [], "shared", "remaining", None, None)
        self.assertEqual(decision["applied"], "read")
        self.assertEqual(trainer.calls, [
            ("shared", "parent", "search"), ("shared", "parent", "read"), ("remaining", "parent:read", "read")])

    def test_each_round_uses_a_disjoint_tail_and_shared_probe_prefix(self):
        lengths = [p.PROBE_STEPS, p.CONTINUE_STEPS] * p.ROUNDS
        parts = train.training_groups(p.MIN_PAIRS, p.SEEDS[0], lengths)
        flat = [i for section in parts for group in section for i in group]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual([len(groups) for groups in parts], lengths)


class BoardTests(unittest.TestCase):
    def test_live_progress_exposes_workers_and_final_go_no_go(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "feedback-test"
            folder.mkdir()
            progress = Progress(folder, "feedback_mechanism")
            progress.update("running", 2, 8, detail="feedback_workers", workers=[
                {"name": "20261123-early-search", "device": "0", "done": 14, "total": 32,
                 "detail": "feedback_probe_search", "metrics": {"train_steps": 14, "other": 9}}])
            phase = board_runs(root, 0)[0]["phases"]["feedback_mechanism"]
            self.assertEqual(phase["workers"][0]["done"], 14)
            self.assertEqual(phase["workers"][0]["metrics"], {"train_steps": 14})
            self.assertEqual(board_runs(root, phase["heartbeat_at"] + 61)[0]["phases"]["feedback_mechanism"]["state"], "stalled")
            progress.update("blocked", 1, 1, detail="feedback_no_go")
            phase = board_runs(root, 0)[0]["phases"]["feedback_mechanism"]
            self.assertEqual((phase["state"], phase["detail"], phase["workers"]), ("blocked", "feedback_no_go", []))


if __name__ == "__main__":
    unittest.main()
