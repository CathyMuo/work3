import contextlib
import gzip
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from work3.amp_gap_math import (G1_MIN_N, analyze, canonical, code_answer, extract_code, gold_answer, last_boxed, vote,
                                wilson_low)


def record(pid, gold, level=3, **arms):
    return {"id": pid, "level": level, "type": "algebra", "gold": gold,
            "arms": {arm: {"answers": answers, "tokens": [10] * len(answers), "finish": ["stop"] * len(answers)}
                     for arm, answers in arms.items()}}


class AnswerTests(unittest.TestCase):
    def test_last_boxed_handles_nesting_and_truncation(self):
        self.assertEqual(last_boxed(r"so \boxed{\frac{1}{2}} done"), r"\frac{1}{2}")
        self.assertEqual(last_boxed(r"\boxed{3} then \boxed{4}"), "4")
        self.assertEqual(last_boxed(r"\boxed{3} then \boxed{\frac{1}{"), "3")
        self.assertEqual(last_boxed(r"answer \boxed 7."), "7.")
        self.assertIsNone(last_boxed("no box"))

    def test_canonical_merges_equivalent_rationals(self):
        for text in (r"\frac{1}{2}", r"\dfrac12", "0.5", "1/2", r"$\frac{1}{2}$", ".5"):
            self.assertEqual(canonical(text), "num:1/2", text)
        self.assertEqual(canonical(r"-\frac{3}{4}"), "num:-3/4")
        self.assertEqual(canonical(r"\frac{-3}{4}"), "num:-3/4")
        self.assertEqual(canonical("1,000"), "num:1000")
        self.assertEqual(canonical("x = 5"), "num:5")
        self.assertEqual(canonical(r"30^\circ"), "num:30")
        self.assertEqual(canonical(r"\text{Monday}"), "tex:Monday")
        self.assertEqual(canonical(r"2\sqrt{3}"), canonical(r"2\sqrt3"))
        self.assertIsNone(canonical("  "))

    def test_code_output_maps_to_the_same_keys(self):
        self.assertEqual(code_answer("debug\n3/4\n"), "num:3/4")
        self.assertEqual(code_answer("0.3333333333333333"), "num:1/3")
        self.assertEqual(code_answer("2*sqrt(3)"), canonical(r"2\sqrt{3}"))
        self.assertIsNone(code_answer("\n  \n"))
        self.assertEqual(extract_code("text\n```python\nprint(1)\n```\nmore\n```python\nprint(2)\n```"), "print(2)")
        self.assertIsNone(extract_code("```python\nprint(1)"))

    def test_gold_from_math_and_gsm8k_rows(self):
        self.assertEqual(gold_answer({"solution": r"Thus $\boxed{\frac{2}{4}}$."}), "num:1/2")
        self.assertEqual(gold_answer({"answer": "steps\n#### 1,234"}), "num:1234")
        self.assertIsNone(gold_answer({"solution": "no box"}))

    def test_vote_breaks_ties_by_first_occurrence_and_counts_missing(self):
        self.assertEqual(vote(["a", "b", "b", "a", None]), ("a", 0.4))
        self.assertEqual(vote([None, None]), (None, 0.0))
        self.assertIsNone(wilson_low(0, 0))
        self.assertLess(wilson_low(9, 10), 0.9)


class GateTests(unittest.TestCase):
    def rows(self, gap_correct):
        """150 easy problems, 40 gap problems, 40 band problems with noisy low labels."""
        rows = []
        for i in range(150):
            rows.append(record(f"e{i}", "num:1", short=["num:1"] * 8, force=["num:1"] * 8))
        for i in range(40):
            high = ["num:2"] * 8 if (i < gap_correct) else ["num:9"] * 8
            rows.append(record(f"g{i}", "num:2", short=["num:2", "num:3", "num:4", "num:5"] + [None] * 4, force=high))
        for i in range(40):
            low = ["num:7"] * 4 + ["num:8"] * 4 if i % 2 else ["num:8"] * 4 + ["num:7"] * 4
            rows.append(record(f"b{i}", "num:7", short=low, force=low))
        return rows

    def test_gap_selection_must_beat_the_band_by_the_margin(self):
        result = analyze(self.rows(gap_correct=38))
        cell = result["gaps"]["force"]["by_tau"]["0.75"]
        self.assertEqual(cell["gap_set"]["n"], 40)
        self.assertEqual(result["gaps"]["force"]["rzero_band"]["n"], 40)
        self.assertTrue(cell["gates"]["G1_calibration"])
        self.assertTrue(cell["gates"]["G2_gap_label_quality"])
        self.assertTrue(cell["gates"]["G3_gap_share"])
        self.assertEqual(result["verdict"], "GO")
        weak = analyze(self.rows(gap_correct=30))
        self.assertFalse(weak["gaps"]["force"]["by_tau"]["0.75"]["gates"]["G2_gap_label_quality"])
        self.assertEqual(weak["verdict"], "NO-GO")

    def test_calibration_needs_enough_confident_problems(self):
        rows = [record(f"x{i}", "num:1", short=["num:1"] * 8, force=["num:1"] * 8) for i in range(G1_MIN_N - 1)]
        cell = analyze(rows)["gaps"]["force"]["by_tau"]["0.75"]
        self.assertFalse(cell["gates"]["G1_calibration"])

    def test_infrastructure_faults_invalidate_an_arm(self):
        rows = self.rows(gap_correct=38)
        for row in rows[:20]:
            row["arms"]["force"]["finish"] = ["infra"] * 8
        result = analyze(rows)
        self.assertFalse(result["gaps"]["force"]["infra_ok"])
        self.assertEqual(result["verdict"], "NO-GO")

    def test_records_must_share_arms(self):
        with self.assertRaises(ValueError):
            analyze([record("a", "num:1", short=["num:1"]), record("b", "num:1", force=["num:1"])])


class DataTests(unittest.TestCase):
    def test_train_files_only_and_balanced_selection(self):
        from work3.amp_gap import select, verify_data

        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "algebra").mkdir()
            train = [{"problem": f"p{i}", "solution": rf"\boxed{{{i}}}", "level": f"Level {1 + i % 5}", "type": "Algebra"}
                     for i in range(60)] + [{"problem": "p0", "solution": r"\boxed{0}", "level": "Level 1"},
                                            {"problem": "nogold", "solution": "none", "level": "Level 2"}]
            (base / "algebra" / "train.jsonl").write_text("\n".join(json.dumps(row) for row in train), encoding="utf-8")
            (base / "algebra" / "test.jsonl").write_text(json.dumps({"problem": "t", "solution": r"\boxed{1}"}), encoding="utf-8")
            report = verify_data(base)
            self.assertEqual(list(report["files"]), ["algebra/train.jsonl"])
            self.assertEqual(report["usable_problems"], 60)
            self.assertEqual(report["dropped"], {"duplicate": 1, "no_gold": 1})
            from work3.amp_gap import load_problems
            problems, _, _ = load_problems(base)
            chosen = select(problems, 5)
            self.assertEqual(len(chosen), 25)
            self.assertEqual(select(problems, 5), chosen)


class SandboxTests(unittest.TestCase):
    def test_outer_deadline_budgets_startup_separately(self):
        from work3 import pair_sandbox

        verifier = pair_sandbox.Verifier.__new__(pair_sandbox.Verifier)
        seen = {}

        class Process:
            pid, returncode = 1, 0

            def communicate(self, data, timeout):
                seen["timeout"] = timeout
                raise pair_sandbox.subprocess.TimeoutExpired("x", timeout)

            def wait(self):
                return 0

        with patch.object(verifier, "command", return_value=["true"]), \
                patch.object(pair_sandbox.subprocess, "Popen", return_value=Process()), \
                patch.object(pair_sandbox.os, "killpg"):
            with self.assertRaisesRegex(RuntimeError, "startup budget"):
                verifier.run_case(0, "", "")
        self.assertEqual(seen["timeout"], pair_sandbox.STARTUP_SECONDS + pair_sandbox.WALL_SECONDS
                         + pair_sandbox.BACKSTOP_SECONDS)
        self.assertEqual(verifier.timing()["calls"], 1)

    def test_limits_are_validated_and_forwarded_to_the_launcher(self):
        from work3 import pair_sandbox

        with self.assertRaises(ValueError):
            pair_sandbox.Verifier(wall=0)
        verifier = pair_sandbox.Verifier.__new__(pair_sandbox.Verifier)
        verifier.mode, verifier.seccomp, verifier.jail, verifier.uid, verifier.gid = "drop-first", True, Path("/j"), 9, 9
        verifier.wall, verifier.cpu, verifier.memory = 6.0, 5, 1 << 30
        self.assertEqual(verifier.command(3)[-4:], ["6.0", "3", "5", str(1 << 30)])

    @unittest.skipUnless(sys.platform == "linux", "Linux namespace sandbox test")
    def test_real_sandbox_runs_math_programs_with_custom_limits(self):
        from work3.amp_gap import CODE_LIMITS, execute
        from work3.pair_sandbox import Verifier

        verifier = Verifier(**CODE_LIMITS)
        self.assertEqual(execute(verifier, "from fractions import Fraction\nprint(Fraction(1, 3) + Fraction(1, 6))"),
                         ("num:1/2", "ok"))
        self.assertEqual(execute(verifier, "raise SystemExit(3)"), (None, "error"))
        self.assertEqual(execute(verifier, None), (None, "no_code"))
        self.assertEqual(execute(verifier, "print('')"), (None, "no_answer"))


class ExecuteRetryTests(unittest.TestCase):
    def test_infrastructure_faults_are_retried_once(self):
        from work3.amp_gap import execute

        class Flaky:
            calls = 0

            def run(self, code):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("startup budget")
                return "ok", "42\n"

        class Broken:
            def run(self, code):
                raise RuntimeError("startup budget")

        with patch("work3.amp_gap.sleep"):
            self.assertEqual(execute(Flaky(), "print(42)"), ("num:42", "ok"))
            self.assertEqual(execute(Broken(), "print(42)"), (None, "infra"))


LIMIT, RESERVED = 100, 140  # stand-ins for Qwen3's 151,669 known ids and a sampled reserved row (feedback #63)


class FakeTokenizer:
    def __init__(self, top):
        self.top = top

    def get_vocab(self):
        return {"a": 0, "b": self.top}


class FakePrompter:
    def __init__(self, model, hybrid):
        self.chat, self.hybrid, self.limit = True, hybrid, LIMIT

    def ids(self, user, system=None, thinking=False):
        return [5, 6, 7]

    def encode(self, text):
        return [8, 9] if text == "0123456789" else [7, 7]

    def stop_ids(self):
        return [0]


class FakeVerifier:
    def __init__(self, **limits):
        pass

    def run(self, code):
        return "ok", "4\n"

    def timing(self):
        return {}


def fake_vllm():
    """A vLLM stand-in that rejects out-of-range prompt ids like 0.8.5 and samples one reserved id per answer."""
    module, made, calls = types.ModuleType("vllm"), [], []

    class SamplingParams:
        def __init__(self, n=1, max_tokens=16, seed=None, **rest):
            self.n, self.max_tokens, self.seed, self.rest = n, max_tokens, seed, rest
            made.append(self)

    class LLM:
        def __init__(self, **options):
            pass

        def generate(self, prompts, sampling_params, use_tqdm=False):
            calls.append(len(prompts))
            results = []
            for prompt, params in zip(prompts, sampling_params, strict=True):
                if max(prompt["prompt_token_ids"]) > LIMIT - 1:
                    raise ValueError(f"Token id {max(prompt['prompt_token_ids'])} is out of vocabulary")
                choices = [types.SimpleNamespace(text="so \\boxed{4}\n```python\nprint(4)\n```",
                                                 token_ids=[3, 4, RESERVED, 0], finish_reason="stop")
                           for _ in range(params.n)]
                results.append(types.SimpleNamespace(outputs=choices))
            return results

    module.LLM, module.SamplingParams = LLM, SamplingParams
    return module, made, calls


class TokenRangeTests(unittest.TestCase):
    def test_limit_is_the_smaller_of_tokenizer_and_embedding(self):
        from work3.amp_gap import token_limit

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(token_limit(FakeTokenizer(151668), tmp), 151669)
            Path(tmp, "config.json").write_text(json.dumps({"vocab_size": 151936}))
            self.assertEqual(token_limit(FakeTokenizer(151668), tmp), 151669)
            Path(tmp, "config.json").write_text(json.dumps({"vocab_size": 1000}))
            self.assertEqual(token_limit(FakeTokenizer(151668), tmp), 1000)

    def test_feedback_filter_and_prompt_guard(self):
        from work3.amp_gap import known, room

        self.assertEqual(known([1, 150, 2, 100], 100), ([1, 2], 2))
        budget = types.SimpleNamespace(max_tokens=10)
        room("force", [[1] * 5], [budget], 15, 100)
        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            room("force", [[1] * 6], [budget], 15, 100)
        with self.assertRaisesRegex(ValueError, "outside the tokenizer range"):
            room("force", [[1, 100]], [budget], 15, 100)

    def test_budgets_fit_the_context_window(self):
        from work3.amp_gap import BUDGETS, MAX_MODEL_LEN, PROMPT_CAP

        for budgets in BUDGETS.values():
            self.assertLessEqual(PROMPT_CAP + budgets["short"] + 64 + budgets["force"], MAX_MODEL_LEN[False])
            self.assertLessEqual(PROMPT_CAP + 128 + budgets["think"], MAX_MODEL_LEN[True])

    def run_worker(self, chunk=64, filtered=True):
        import work3.amp_gap as amp

        module, made, calls = fake_vllm()
        problems = [{"id": f"p{i}", "problem": f"What is {i}?", "level": 1 + i, "type": "Algebra", "gold": "num:4"}
                    for i in range(3)]
        settings = {"gpus": 1, "samples": 2, "budgets": amp.BUDGETS["smoke"], "hybrid": False, "max_model_len": 8192,
                    "gpu_memory": 0.5, "token_limit": LIMIT}
        unfiltered = patch.object(amp, "known", lambda tokens, limit: (tokens, 0))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"vllm": module}), \
                patch.object(amp, "Prompter", FakePrompter), patch.object(amp, "Verifier", FakeVerifier), \
                patch.object(amp, "CHUNK", chunk), unfiltered if not filtered else contextlib.nullcontext(), \
                contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            (directory / "plan.json").write_text(json.dumps({"model": "/fake", "settings": settings, "problems": problems}))
            amp.worker(types.SimpleNamespace(directory=directory, rank=0))
            rows = [json.loads(line) for line in (directory / "rank-0.jsonl").read_text().splitlines()]
            status = json.loads((directory / "status-0.json").read_text())
        return rows, status, made, calls

    def test_sampled_reserved_ids_never_reach_a_prompt(self):
        rows, status, made, _ = self.run_worker()
        self.assertFalse(any("allowed_token_ids" in params.rest for params in made))  # sampling stays as frozen
        self.assertEqual([row["oov_dropped"] for row in rows], [2, 2, 2])
        self.assertTrue(all(row["oov_tokens"] == 2 * 4 for row in rows))  # one per sample in four arms
        self.assertEqual((status["done"], status["oov_dropped"]), (3, 6))
        self.assertEqual(rows[0]["arms"]["force"]["answers"], ["num:4", "num:4"])

    def test_without_the_filter_the_force_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "force: prompt token id 140 is outside the tokenizer range"):
            self.run_worker(filtered=False)

    def test_independent_arms_share_one_call_and_chunks_pipeline(self):
        _, _, _, calls = self.run_worker()
        self.assertEqual(calls, [3 + 3, 3 * 2 + 3 * 2])  # short+code prompts, then force+verify per sample
        rows, status, _, calls = self.run_worker(chunk=1)
        self.assertEqual([row["id"] for row in rows], ["p0", "p1", "p2"])
        self.assertEqual((len(calls), status["done"]), (6, 3))
        self.assertTrue(all(row["arms"]["code"]["finish"] == ["ok", "ok"] for row in rows))

if __name__ == "__main__":
    unittest.main()
