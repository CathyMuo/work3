"""Exploratory diagnostics added after P0 (factory feedback #10, #11): eval-dump behaviour,
gold-answer retrieval, the micro-batch ablation comparison, and P1 by hop."""
import json
import tempfile
import unittest
from pathlib import Path

from work3 import p0_diag as D, p1_summary as S

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


def call(n_queries=1):
    return '<tool_call>{"name": "search", "arguments": {"query_list": %s}}</tool_call>' % json.dumps(["q"] * n_queries)


def response(docs):
    return "".join("user\n<tool_response>%s</tool_response>\n" % json.dumps({"result": d}) for d in docs)


def write_dump(d: Path, rows):
    """rows: (dataset, question, score, searches, queries_per_call, retrieved docs)"""
    d.mkdir(parents=True)
    with (d / "0.jsonl").open("w") as f:
        for ds, q, score, searches, qpc, docs in rows:
            out = "<think>x</think>" + (call(qpc) * searches) + response(docs) + "<answer>a</answer>"
            prompt = f"system\nAnswer. Question: an example.\nuser\nUse search. Question: {q}\nassistant\n"
            f.write(json.dumps({"input": prompt, "output": out, "score": score,
                                "data_source": "searchR1_" + ds}) + "\n")


class P0DiagTest(unittest.TestCase):
    def test_counts_pairs_and_compare(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_dump(root / "iter1", [("hotpotqa", "q1", 1.0, 2, 1, []), ("hotpotqa", "q2", 1.0, 2, 1, []),
                                        ("nq", "q3", 0.0, 1, 1, [])])
            write_dump(root / "iter3", [("hotpotqa", "q2", 0.0, 1, 2, []), ("hotpotqa", "q1", 0.0, 0, 1, []),
                                        ("nq", "q3", 1.0, 1, 1, [])])
            write_dump(root / "abl", [("hotpotqa", "q1", 1.0, 2, 1, []), ("hotpotqa", "q2", 0.0, 1, 1, []),
                                      ("nq", "q3", 1.0, 1, 1, [])])
            r = D.diagnose(root, compare={"micro2": root / "abl"})
            m1 = r["per_run"]["1"]["multi_hop"]
            self.assertEqual((m1["n"], m1["em"], m1["searches_mean"], m1["queries_per_call"]), (2, 1.0, 2.0, 1.0))
            m3 = r["per_run"]["3"]["multi_hop"]
            self.assertEqual((m3["no_search_share"], m3["queries_mean"]), (0.5, 1.0))
            self.assertEqual(m3["queries_per_call"], 2.0)          # one call carrying two queries
            p = r["paired"]["1->3"]["multi_hop"]
            self.assertEqual((p["pairs"], p["correct_to_wrong"], p["wrong_to_correct"]), (2, 2, 0))
            self.assertEqual(p["correct_to_wrong_by_search_change"], {"fewer": 2, "same": 0, "more": 0})
            self.assertAlmostEqual(p["searches_change_mean"], -1.5)
            self.assertEqual(r["paired"]["3->micro2"]["multi_hop"]["wrong_to_correct"], 1)
            self.assertIn("multi_hop", D.table(r))

    def test_question_text_and_duplicates(self):
        self.assertEqual(D.question_of("x Question: example\nuser\nQuestion: who is it?\nassistant\n"), "who is it?")
        with tempfile.TemporaryDirectory() as d:
            write_dump(Path(d) / "iter1", [("popqa", "same", 1.0, 1, 1, []), ("popqa", "same", 0.0, 1, 1, []),
                                           ("popqa", "other", 1.0, 1, 1, [])])
            rows = D.load(Path(d) / "iter1", {})
            self.assertEqual([k[1] for k in rows], ["other"])

    @unittest.skipUnless(HAVE_PANDAS, "pandas not installed")
    def test_gold_retrieval_split(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            test = root / "test.parquet"
            pd.DataFrame({"data_source": ["searchR1_hotpotqa"] * 3,
                          "extra_info": [{"question": q} for q in ("q1", "q2", "q3")],
                          "reward_model": [{"ground_truth": {"target": ["The Beatles"]}}] * 3}).to_parquet(test)
            write_dump(root / "iter1", [("hotpotqa", "q1", 1.0, 1, 1, ["... the beatles were ..."]),
                                        ("hotpotqa", "q2", 0.0, 1, 1, ["nothing relevant"]),
                                        ("hotpotqa", "q3", 0.0, 2, 1, ["x", "Beatles, The"])])
            s = D.diagnose(root, str(test))["per_run"]["1"]["multi_hop"]
            self.assertEqual(s["gold_known"], 3)
            self.assertAlmostEqual(s["gold_retrieved_share"], 2 / 3)   # "Beatles, The" normalises to "beatles"
            self.assertAlmostEqual(s["wrong_gold_never_retrieved"], 1 / 3)
            self.assertAlmostEqual(s["wrong_gold_retrieved"], 1 / 3)

    def test_invalid_tool_calls_do_not_count(self):
        self.assertEqual(D.valid_searches(call() + "<tool_call>not json</tool_call>"
                                          '<tool_call>{"name": "other"}</tool_call>'), 1)
        self.assertEqual(D.search_calls(call(3) + call(1)), [3, 1])

    def test_mcnemar(self):
        self.assertIsNone(D.mcnemar_p(0, 0))
        self.assertAlmostEqual(D.mcnemar_p(5, 5), 1.0)
        self.assertAlmostEqual(D.mcnemar_p(0, 10), 2 / 1024)
        self.assertLess(D.mcnemar_p(300, 200), 1e-4)


class P1ByHopTest(unittest.TestCase):
    def test_hop_groups_and_eta(self):
        self.assertEqual([S.hop_group(x) for x in ("search_zero_1", "search_zero_2", "search_zero_4", None)],
                         ["1", "2", "3+", None])
        recs = [{"qid": q, "scores": [1, 0, 0, 0, 0], "data_source": "search_zero_3",
                 "replays": [[{"score": 1}, {"score": 0}]] + [[{"score": 0}, {"score": 0}]] * 4}
                for q in range(20)]
        h = S.by_hop([S.question_stats(r) for r in recs], boot=50)
        self.assertEqual(h["3+"]["questions"], 20)
        self.assertAlmostEqual(h["3+"]["eta_k1"], (0.5 / 5) / 0.2)
        self.assertEqual(h["1"]["questions"], 0)
        self.assertIsNotNone(h["3+"]["eta_mixed_ci95"])


if __name__ == "__main__":
    unittest.main()
