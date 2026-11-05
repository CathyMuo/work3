import json
import tempfile
import unittest
from pathlib import Path

from work3 import p0_gate


def write_dump(directory, em_by_dataset, n=20, extra_rows=()):
    """Write a synthetic verl validation dump with the given per-dataset EM."""
    path = Path(directory) / "0.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for name, em in em_by_dataset.items():
            n_correct = round(em * n)
            for i in range(n):
                row = {"input": "q", "output": "a", "step": 0,
                       "data_source": "searchR1_" + name,
                       "score": 1.0 if i < n_correct else 0.0}
                f.write(json.dumps(row) + "\n")
        for row in extra_rows:
            f.write(json.dumps(row) + "\n")
    return path


def uniform(em):
    return {d: em for d in p0_gate.DATASETS}


class SummarizeTest(unittest.TestCase):
    def test_average_is_unweighted_mean_of_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            ems = dict(zip(p0_gate.DATASETS, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
            s = p0_gate.summarize_dump(write_dump(d, ems))
            self.assertAlmostEqual(s["average"], 0.4)
            self.assertEqual(s["problems"], [])
            self.assertFalse(s["counts_match_flashrag"])

    def test_directory_with_one_dump_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            write_dump(d, uniform(0.5))
            self.assertAlmostEqual(p0_gate.summarize_dump(d)["average"], 0.5)

    def test_directory_with_two_dumps_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_dump(d, uniform(0.5))
            (Path(d) / "1.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                p0_gate.summarize_dump(d)

    def test_missing_dataset_and_nonbinary_scores_are_problems(self):
        with tempfile.TemporaryDirectory() as d:
            ems = uniform(0.5)
            del ems["bamboogle"]
            bad = {"data_source": "searchR1_nq", "score": 0.5}
            s = p0_gate.summarize_dump(write_dump(d, ems, extra_rows=[bad]))
            self.assertIsNone(s["average"])
            self.assertEqual(len(s["problems"]), 2)

    def test_unknown_sources_are_counted_not_scored(self):
        with tempfile.TemporaryDirectory() as d:
            extra = [{"data_source": "search_zero_1", "score": 1.0}]
            s = p0_gate.summarize_dump(write_dump(d, uniform(0.0), extra_rows=extra))
            self.assertEqual(s["other_sources"], {"search_zero_1": 1})
            self.assertEqual(s["average"], 0.0)


class DecideTest(unittest.TestCase):
    def summary(self, em):
        with tempfile.TemporaryDirectory() as d:
            return p0_gate.summarize_dump(write_dump(d, uniform(em), n=1000))

    def test_threshold_is_frozen_value(self):
        self.assertEqual(p0_gate.THRESHOLD, 0.306)

    def test_pass_at_threshold(self):
        r = p0_gate.decide({3: self.summary(0.306)})
        self.assertEqual(r["verdict"], "PASS")

    def test_fail_below_threshold(self):
        r = p0_gate.decide({3: self.summary(0.305)})
        self.assertEqual(r["verdict"], "FAIL")

    def test_missing_iter3_is_invalid(self):
        r = p0_gate.decide({1: self.summary(0.9)})
        self.assertEqual(r["verdict"], "INVALID")

    def test_problem_in_any_iteration_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            ems = uniform(0.5)
            del ems["nq"]
            broken = p0_gate.summarize_dump(write_dump(d, ems))
        r = p0_gate.decide({1: broken, 3: self.summary(0.5)})
        self.assertEqual(r["verdict"], "INVALID")

    def test_failed_pool_check_is_invalid(self):
        r = p0_gate.decide({3: self.summary(0.5)}, pool_check_ok=False)
        self.assertEqual(r["verdict"], "INVALID")

    def test_delta_vs_paper(self):
        r = p0_gate.decide({3: self.summary(0.326)})
        self.assertAlmostEqual(r["delta_vs_paper"]["3"]["nq"], 0.326 - 0.397, places=6)

    def test_cli_exit_code_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            dump = write_dump(d, uniform(0.4))
            out = Path(d) / "s.json"
            code = p0_gate.main(["--iter3", str(dump), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
