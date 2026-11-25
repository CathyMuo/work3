import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from time import time
from unittest.mock import patch

from work3.feedback_pair import initial_row, join, plan_repairs
from work3.feedback_pair_data import DATA_REVISION, Verifier, clean_code, similar_input, split_tests, verify_source
from work3.feedback_pair_metrics import BRANCHES, PAIRS, REPEATS, choose, collect_scored, fit
from work3.pair_sandbox import SandboxStartError
from work3.wiki_rsi_progress import Progress, board_runs


class Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"].split()


def examples(pairs):
    return {"input": [a for a, _ in pairs], "output": [b for _, b in pairs]}


def trace(success):
    return {"success": success, "tokens": 100, "prompt_tokens": 60, "output_tokens": 40,
            "executions": 8, "judge_seconds": 0.1, "generation_seconds": 0.2, "retained": 1, "repair_fraction": 0.5}


class FeedbackPairTests(unittest.TestCase):
    def test_feedback_and_score_exclude_neighbors_and_conflicts(self):
        row = {"public_tests": examples([("1 2 3", "4")]),
               "generated_tests": examples([("9 2 6", "10"), ("1 2 4", "5")]),
               "private_tests": examples([("9 2 7", "2"), ("3 4 5", "9")])}
        feedback, score = split_tests(row)
        self.assertEqual(feedback, [("9 2 6", "10")])
        self.assertEqual(score, [("3 4 5", "9")])
        self.assertTrue(similar_input("1 2 3\n", " 1  2  4"))
        row["private_tests"] = examples([("1  2  3", "12")])
        self.assertEqual(split_tests(row), ([], []))

    def test_only_visible_failures_trigger_branches(self):
        class FakeVerifier:
            def evaluate(self, code, cases):
                return [("wrong" if expected == "x" else "pass", "a") for _, expected in cases], len(cases), 0

        task = {"id": "example", "contest": 2, "split": "sealed", "description": "Echo an element",
                "feedback": [("1", "x"), ("2", "x"), ("3", "a"), ("4", "a")], "score": [("5", "a")]}
        reply = ("print('a')", 80, 8, 0.1, False)
        first = initial_row(task, reply, Tokenizer(), FakeVerifier())
        self.assertTrue(first["initial_correct"])
        self.assertTrue(first["eligible"])
        self.assertEqual(len(plan_repairs([task], [first])[0]), 1)
        task["feedback"] = [("1", "a")]
        task["score"] = [("5", "x")]
        first = initial_row(task, reply, Tokenizer(), FakeVerifier())
        self.assertFalse(first["initial_correct"])
        self.assertEqual(plan_repairs([task], [first])[0], [])

    def test_missing_or_duplicate_repair_branch_is_rejected(self):
        state = {"id": "p", "split": "train", "eligible": True}
        records = [{"id": "p", "arm": arm, "repeat": repeat}
                   for arm in BRANCHES for repeat in range(REPEATS)]
        self.assertEqual(set(join([(None, state)], records)["train"][0]["branches"]), set(BRANCHES))
        with self.assertRaises(KeyError):
            join([(None, state)], records[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            join([(None, state)], records + records[:1])

    def test_models_share_features_and_fallback_row_has_zero_gain(self):
        row = {"id": "yes", "contest": 1, "eligible": True, "closest": PAIRS[0],
               "features": [[0], [1], [2], [4]],
               "branches": {arm: [trace(arm in {PAIRS[0], PAIRS[3]}) for _ in range(REPEATS)] for arm in BRANCHES}}
        fallback = {"id": "other", "contest": 2, "eligible": False,
                    "branches": {"regenerate": [trace(True) for _ in range(REPEATS)]}}
        outcome, crossed = fit([row], 0), fit([row], 2)
        self.assertEqual(outcome["center"], crossed["center"])
        self.assertNotEqual(outcome["coef"], crossed["coef"])
        self.assertIn(choose(crossed, row["features"]), PAIRS)
        summary, models, reports = collect_scored([row], [row, fallback])
        self.assertEqual(summary["verdict"], "NO-GO")
        self.assertEqual(reports[-1]["interaction"], reports[-1]["outcome_only"])
        self.assertEqual(len(models), 2)

    def test_experiment_progress_includes_worker_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair-repair-test"
            path.mkdir()
            progress = Progress(path, "pair_repair")
            progress.update("running", 4, 8, detail="pair_repair",
                            workers=[{"name": "rank-0", "device": "0", "done": 1, "total": 2}])
            row = board_runs(Path(directory), time())[0]["phases"]["pair_repair"]
            self.assertEqual((row["done"], row["total"], row["workers"][0]["done"]), (4, 8, 1))

    def test_complete_python_only(self):
        self.assertEqual(clean_code("```python\nprint(1)\n```"), "print(1)")
        self.assertIsNone(clean_code("```python\nprint("))

    def test_failure_to_start_isolation_is_not_a_wrong_answer(self):
        verifier = Verifier.__new__(Verifier)
        with patch.object(verifier, "command", return_value=[sys.executable, "-c", "import sys; sys.exit(1)"]):
            r_fd, w_fd = __import__("os").pipe()
            try:
                __import__("os").close(w_fd)
                with self.assertRaisesRegex(SandboxStartError, "isolated Python did not start"):
                    verifier.run_case(r_fd, "1", "2")
            finally:
                __import__("os").close(r_fd)

    def test_probe_rejects_broken_isolation_and_leftovers(self):
        verifier = Verifier.__new__(Verifier)
        verifier.uid, verifier.gid, verifier.mode, verifier.seccomp = 65534, 65534, "drop-first", True
        with patch("work3.pair_sandbox.os.geteuid", return_value=0, create=True), \
                patch.object(verifier, "uid_pids", return_value=set()), \
                patch.object(verifier, "evaluate", return_value=([("wrong", "fork was allowed")], 1, 0)):
            with self.assertRaisesRegex(RuntimeError, "fork was allowed"):
                verifier.probe()
        # Pre-existing uid processes are ignored; newly leaked processes fail probe.
        calls = iter([([("pass", "work3-isolation")], 1, 0.1), ([("pass", "nproc-ok")], 1, 0.1),
                      ([("timeout", "")], 1, 2.5), ([("error", "")], 1, 1.0)])
        pid_snapshots = iter([{42}, {42, 999}] + [{42, 999}] * 25)
        with patch("work3.pair_sandbox.os.geteuid", return_value=0, create=True), \
                patch("work3.pair_sandbox.sleep"), \
                patch.object(verifier, "uid_pids", side_effect=lambda: next(pid_snapshots)), \
                patch.object(verifier, "evaluate", side_effect=lambda *a, **k: next(calls)):
            with self.assertRaisesRegex(RuntimeError, "999"):
                verifier.probe()

    @unittest.skipUnless(sys.platform == "linux", "Linux namespace sandbox test")
    def test_real_linux_sandbox_blocks_fork_threads_and_writes(self):
        verifier = Verifier()
        report = verifier.probe()
        self.assertEqual(report["isolation"], "pass")
        self.assertEqual(report["nproc_independent"], "pass")
        self.assertEqual(report["sleep_limit"], "timeout")
        [(status, out)], count, _ = verifier.evaluate("a, b = map(int, input().split()); print(a + b)", [("3 4\n", "7")])
        self.assertEqual((status, out, count), ("pass", "7", 1))
        [(err_status, _)], _, _ = verifier.evaluate("import os; os.fork()", [("", "")])
        self.assertEqual(err_status, "error")
        [(ro_status, _)], _, _ = verifier.evaluate("open('/escape', 'w').write('x')", [("", "")])
        self.assertEqual(ro_status, "error")

    def test_verify_source_pins_content_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            files = {f"train-{i:05d}.parquet": {"bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
                     for i in range(39)}
            manifest = root / "lock.json"
            manifest.write_text(json.dumps({"revision": DATA_REVISION, "shards": files}), encoding="utf-8")
            with patch("work3.feedback_pair_data.SHARD_MANIFEST", manifest):
                with self.assertRaisesRegex(ValueError, "0/39"):
                    verify_source(root)
                for name in files:
                    (root / "data" / name).write_bytes(b"x")
                self.assertEqual(verify_source(root)["verified_shards"], 39)
                (root / "data" / next(iter(files))).write_bytes(b"y")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    verify_source(root)


if __name__ == "__main__":
    unittest.main()
