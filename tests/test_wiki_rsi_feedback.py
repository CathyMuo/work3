"""Regression tests for automatic gates, branch reuse and live board snapshots."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from work3 import wiki_rsi_feedback as runner
from work3 import wiki_rsi_feedback_data as data
from work3 import wiki_rsi_feedback_protocol as p
from work3 import wiki_rsi_feedback_train as train
from work3.evidence_ref import sha256
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
        self.assertFalse(p.enough_screen_pairs({"person": 29, "place": 30, "work": 20}))
        self.assertFalse(p.enough_screen_pairs({"person": 60, "place": 40, "work": 9}))
        self.assertTrue(p.enough_screen_pairs({"person": 40, "place": 30, "work": 10}))

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

    def test_zero_pair_v1_import_preserves_frozen_pages_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            v1, v2 = (Path(temporary) / stage for stage in ("v1", "v2"))
            v1.mkdir()
            v2.mkdir()
            runtime = {"python": "test"}
            locked = {"protocol": p.LEGACY_PROTOCOL, "titles": ["one", "two"], "source": {"corpus": "abc"},
                      "models": {"early": {}, "late": {}}, "splits": {"confirmation": [{"page": "sealed"}]}}
            original = v1 / data.LOCKED
            atomic_json(original, locked)
            common = {"protocol": p.LEGACY_PROTOCOL, "repo_revision": p.LEGACY_RELEASE,
                      "contract": {"test": "abc"}, "runtime_versions": runtime}
            atomic_json(runner.receipt_path(v1, "lock"), {**common, "phase": "lock", "verdict": "GO",
                        "locked_sha256": sha256(original), "models": locked["models"], "inputs": locked["source"],
                        "shared_first_queries": 1, "by_split": {}, "clusters": {}, "sources": 2, "drzero_revision": "abc"})
            atomic_json(runner.receipt_path(v1, "prepare"), {**common, "phase": "prepare", "verdict": "NO-GO",
                        "count": 0, "pairs_sha256": None})
            args = SimpleNamespace(import_v1=v1, early_model=None, late_model=None)
            with patch.object(p, "PROPOSALS", 2), patch.object(runner, "verify_context", return_value={}):
                result = runner.reuse_v1_lock(args, v2, Progress(v2, "feedback_lock"), runtime)
                self.assertTrue(result["ready"])
                self.assertEqual(runner.read_json(v2 / data.LOCKED), {**locked, "protocol": p.PROTOCOL})
                self.assertEqual(runner.read_json(original), locked)
                atomic_json(original, {**locked, "titles": ["changed", "two"]})
                with self.assertRaises(ValueError):
                    runner.reuse_v1_lock(args, v2, Progress(v2, "feedback_lock"), runtime)

    def test_prepare_no_go_never_opens_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            worker = {"sources": 128, "funnel": {}, "pairs": [], "queries": 130}
            with patch.object(runner, "run_workers", return_value=("screen-dir", {"source-screen-0": worker})) as run:
                result = runner.prepare(folder, Progress(folder, "feedback_prepare"), 1e15,
                                        {"blocked_train_pages": []}, {}, {}, ["0"], Path("test"))
                self.assertFalse(result["ready"])
                self.assertEqual(result["reason"], "screen_no_go")
                self.assertEqual(run.call_count, 1)
                self.assertFalse((folder / data.PAIRS).exists())


class DataTests(unittest.TestCase):
    def test_compact_proof_ids_recover_original_verified_text(self):
        source, bridge = "Sample Film", "Example Person"
        first = [(source, f"{source} was directed by {bridge}.")]
        second = [(bridge, f"{bridge} was born in Sampletown.")]
        task = (source, bridge, first, second, [first[0][1]], [second[0][1]], "person")
        funnel = Counter()
        valid = '{"question":"Where was the director of Sample Film born?","answer":"Sampletown","proof":[1,1]}'
        result = data.decode_pair(valid, task, funnel)
        self.assertEqual(result["answer"], "Sampletown")
        self.assertIsNone(data.decode_pair(valid.replace("[1,1]", "[1,2]"), task, funnel))
        self.assertIsNone(data.decode_pair(valid.replace("Sampletown", "Another City"), task, funnel))
        self.assertEqual(funnel, {"invalid_span_ids": 1, "semantic_or_evidence_rejected": 1})

    def test_candidates_exclude_the_source_and_use_real_bounded_spans(self):
        body = ("Sample Film was directed by Example Person. Old Studio produced it. "
                "Bridge evidence includes " + "real words " * 36 + "entirely within this retrieved passage.")
        options = dict(data.bridge_options("Sample Film", body, {"old studio"}))
        self.assertNotIn("Sample Film", options)
        self.assertNotIn("Old Studio", options)
        self.assertIn("Example Person", options)
        for span in data.text_spans(body):
            self.assertTrue(12 <= len(span) <= 280)
            self.assertIn(span, body)

    def test_merge_uses_a_rare_alternative_before_reused_support(self):
        candidates = [{"qid": "a1", "source": "Source A", "bridge": "Frequent Page", "first": [], "second": []},
                      {"qid": "a2", "source": "Source A", "bridge": "Rare Page", "first": [], "second": []},
                      {"qid": "b1", "source": "Source B", "bridge": "Frequent Page", "first": [], "second": []}]
        rows = data.merge_pairs([{"pairs": candidates}], {"blocked_train_pages": []})
        self.assertEqual([row["qid"] for row in rows], ["a2", "b1"])

    def test_prepare_shard_uses_real_returned_path_with_compact_generation(self):
        question = "Where was the director of Sample Film born?"
        first = [("Sample Film", "Sample Film was directed by Example Person.")]
        second = [("Example Person", "Example Person was born in Sampletown.")]
        returns = {"Sample Film": first, "Example Person": second, question: first}
        retriever = SimpleNamespace(search=lambda queries: [returns.get(value, []) for value in queries])
        result = {"text": '{"question":"' + question + '","answer":"Sampletown","proof":[1,1]}',
                  "finish": "stop"}
        spec = {"retriever": "unused", "source_range": [0, 1], "shard": 0}
        locked = {"titles": ["Sample Film"], "blocked_train_pages": [], "splits": {}}
        with patch.object(data.wiki, "Retriever", return_value=retriever), patch.object(data, "synthesis_prompt"):
            report = data.prepare_shard(spec, locked, None, None, None, lambda *args: [result], lambda *a, **k: None)
            pair, = report["pairs"]
            self.assertEqual((pair["question"], pair["bridge"]), (question, "Example Person"))
            self.assertTrue(pair["question_first_witness"])
            self.assertEqual(report["queries"], 3)
            self.assertEqual(report["funnel"]["bridge_sources_retrieved"], 1)

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
