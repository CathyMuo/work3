import gzip
import json
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
