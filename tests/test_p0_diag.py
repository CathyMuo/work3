"""Exploratory diagnostics added after P0 (factory feedback #10): eval-dump behaviour and P1 by hop."""
import json
import tempfile
import unittest
from pathlib import Path

from work3 import p0_diag as D, p1_summary as S

CALL = '<tool_call>{"name": "search", "arguments": {"query_list": ["q"]}}</tool_call>'


def write_dump(d: Path, rows):
    d.mkdir(parents=True)
    with (d / "0.jsonl").open("w") as f:
        for ds, q, score, searches in rows:
            out = "<think>x</think>" + CALL * searches + "<answer>a</answer>"
            f.write(json.dumps({"input": q, "output": out, "score": score,
                                "data_source": "searchR1_" + ds}) + "\n")


class P0DiagTest(unittest.TestCase):
    def test_counts_and_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_dump(root / "iter1", [("hotpotqa", "q1", 1.0, 2), ("hotpotqa", "q2", 1.0, 2),
                                        ("nq", "q3", 0.0, 1)])
            write_dump(root / "iter3", [("hotpotqa", "q2", 0.0, 1), ("hotpotqa", "q1", 0.0, 0),
                                        ("nq", "q3", 1.0, 1)])
            r = D.diagnose(root)
            m1 = r["per_iteration"]["1"]["multi_hop"]
            self.assertEqual((m1["n"], m1["em"], m1["searches_mean"]), (2, 1.0, 2.0))
            m3 = r["per_iteration"]["3"]["multi_hop"]
            self.assertEqual((m3["no_search_share"], m3["em_by_searches"]["0"]["share"]), (0.5, 0.5))
            p = r["paired"]["1->3"]["multi_hop"]
            self.assertEqual((p["pairs"], p["correct_to_wrong"], p["wrong_to_correct"]), (2, 2, 0))
            self.assertAlmostEqual(p["searches_change_mean"], -1.5)
            self.assertEqual(r["paired"]["1->3"]["single_hop"]["wrong_to_correct"], 1)
            self.assertIn("multi_hop", D.table(r))

    def test_invalid_tool_calls_do_not_count(self):
        self.assertEqual(D.valid_searches(CALL + "<tool_call>not json</tool_call>"
                                          '<tool_call>{"name": "other"}</tool_call>'), 1)

    def test_mcnemar(self):
        self.assertIsNone(D.mcnemar_p(0, 0))
        self.assertAlmostEqual(D.mcnemar_p(5, 5), 1.0)
        self.assertAlmostEqual(D.mcnemar_p(0, 10), 2 / 1024)
        self.assertLess(D.mcnemar_p(300, 200), 1e-4)


class P1ByHopTest(unittest.TestCase):
    def test_hop_groups_and_eta(self):
        self.assertEqual([S.hop_group(x) for x in ("search_zero_1", "search_zero_2", "search_zero_4", None)],
                         ["1", "2", "3+", None])
        recs = []
        for q in range(20):
            r = {"qid": q, "scores": [1, 0, 0, 0, 0], "data_source": "search_zero_3",
                 "replays": [[{"score": 1}, {"score": 0}]] + [[{"score": 0}, {"score": 0}]] * 4}
            recs.append(r)
        st = [S.question_stats(r) for r in recs]
        h = S.by_hop(st, boot=50)
        self.assertEqual(h["3+"]["questions"], 20)
        self.assertAlmostEqual(h["3+"]["eta_k1"], (0.5 / 5) / 0.2)
        self.assertEqual(h["1"]["questions"], 0)
        self.assertIsNotNone(h["3+"]["eta_mixed_ci95"])


if __name__ == "__main__":
    unittest.main()
