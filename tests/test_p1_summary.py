import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from work3 import p1_agent as A, p1_run, p1_summary as S
from test_p1_agent import FakeTokenizer, TOOL


def record(qid, k, noisy_pairs, new_search=0, pool_id=1):
    """k successes; `noisy_pairs` of the 5 replay pairs disagree."""
    scores = [1] * k + [0] * (5 - k)
    replays = []
    for i in range(5):
        a, b = (1, 0) if i < noisy_pairs else (scores[i], scores[i])
        replays.append([{"replayable": True, "score": a, "new_tool_calls": 1 if (i == 0 and new_search) else 0},
                        {"replayable": True, "score": b, "new_tool_calls": 0}])
    return {"pool_id": pool_id, "qid": qid, "scores": scores, "k": k, "replays": replays,
            "originals": [{"finish": "stop"}] * 5}


def write_run(d, pools, lucky_succ=0):
    d = Path(d)
    for p, recs in pools.items():
        with (d / f"pool{p}_questions.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with (d / f"pool{p}_lucky.jsonl").open("w") as f:
            for r in recs:
                if r["k"] == 1:
                    f.write(json.dumps({"qid": r["qid"], "scores": [1] * lucky_succ + [0] * (16 - lucky_succ)}) + "\n")
        (d / f"pool{p}_state.json").write_text(json.dumps(
            {"batches": [{"seed": 1, "qids": [r["qid"] for r in recs]}]}))


def pools_with(noisy_k1, noisy_k34, n=70, new_search=0):
    out = {}
    for p in (1, 3):
        recs = []
        for q in range(n):
            recs.append(record(q, 1, noisy_k1, new_search=new_search, pool_id=p))
        for q in range(n, 2 * n):
            recs.append(record(q, 3, noisy_k34, pool_id=p))
        recs.append(record(2 * n, 0, 0, pool_id=p))
        out[p] = recs
    return out


class StatsTest(unittest.TestCase):
    def test_question_stats(self):
        s = S.question_stats(record(0, 1, 2))
        self.assertAlmostEqual(s["tot"], 1 * 4 / 20)
        self.assertAlmostEqual(s["noise"], (2 * 0.5) / 5)
        self.assertEqual(s["replays"], 10)

    def test_eta_ratio_of_sums_skips_degenerate(self):
        st = [S.question_stats(record(0, 1, 2)), S.question_stats(record(1, 0, 0))]
        self.assertAlmostEqual(S.eta(st, (1,)), 0.2 / 0.2)
        self.assertIsNone(S.eta(st, (3, 4)))


class GateTest(unittest.TestCase):
    def run_summary(self, pools, pool_ok=True, lucky_succ=0):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, pools, lucky_succ)
            return S.summarize(Path(d), pool_ok, None)

    def test_go(self):
        # k=1: tot 0.2, noise 2*0.5/5 = 0.2 -> eta 1.0 ; k=3: tot 0.3, noise 0 -> 0
        r = self.run_summary(pools_with(2, 0))
        self.assertEqual(r["verdict"], "GO", r["problems"])
        self.assertAlmostEqual(r["pooled"]["eta_k1"], 1.0)
        self.assertAlmostEqual(r["pooled"]["diff"], 1.0)
        self.assertTrue(r["secondary"]["d_lucky_share_ge_0.30"])

    def test_no_go_when_groups_equally_noisy(self):
        r = self.run_summary(pools_with(1, 2))   # eta(k=1) 0.5, eta(k=3) 0.67
        self.assertEqual(r["verdict"], "NO-GO")
        self.assertTrue(r["primary"]["a_eta_k1_ge_0.35"])
        self.assertFalse(r["primary"]["b_diff_ge_0.10_ci_above_0"])

    def test_no_go_when_noise_small(self):
        r = self.run_summary(pools_with(0, 0))
        self.assertEqual(r["verdict"], "NO-GO")

    def test_invalid_small_groups(self):
        r = self.run_summary(pools_with(2, 0, n=30))
        self.assertEqual(r["verdict"], "INVALID")

    def test_invalid_new_search_rate(self):
        r = self.run_summary(pools_with(2, 0, new_search=1))   # 70/(141*10) per pool ~ 5%
        self.assertEqual(r["verdict"], "GO")
        pools = pools_with(2, 0)
        for recs in pools.values():
            for rec in recs:
                for pair in rec["replays"]:
                    for rep in pair:
                        rep["new_tool_calls"] = 1
        self.assertEqual(self.run_summary(pools)["verdict"], "INVALID")

    def test_invalid_without_pool_check(self):
        self.assertEqual(self.run_summary(pools_with(2, 0), pool_ok=None)["verdict"], "INVALID")

    def test_bootstrap_deterministic(self):
        pools = {p: [S.question_stats(r) for r in recs] for p, recs in pools_with(2, 1).items()}
        self.assertEqual(S.bootstrap(pools, boot=200), S.bootstrap(pools, boot=200))


class RunHelpersTest(unittest.TestCase):
    def test_draw_batch_disjoint_and_seeded(self):
        qids = list(range(2000))
        b1 = p1_run.draw_batch(qids, set(), 20261106)
        b2 = p1_run.draw_batch(qids, set(b1), 20261107)
        self.assertEqual(len(b1), 500); self.assertFalse(set(b1) & set(b2))
        self.assertEqual(b1, p1_run.draw_batch(qids, set(), 20261106))
        self.assertEqual(len(p1_run.draw_batch(list(range(10)), set(), 1)), 10)

    def test_pool_rows_filters_empty(self):
        import pandas as pd
        prefix = "Answer ... For example, <answer> Beijing </answer>. Question: "
        df = pd.DataFrame({
            "prompt": [[{"role": "user", "content": prefix + "Who?"}],
                       [{"role": "user", "content": prefix + "  "}],
                       [{"role": "user", "content": prefix + "Where?"}]],
            "reward_model": [{"ground_truth": {"target": ["A"]}}, {"ground_truth": {"target": ["B"]}},
                             {"ground_truth": {"target": [""]}}],
            "data_source": ["search_zero_1"] * 3})
        rows = p1_run.pool_rows(df)
        self.assertEqual([r["qid"] for r in rows], [0])
        self.assertEqual(rows[0]["question"], "Who?")

    def test_run_question_structure(self):
        class FakeClients:
            def __init__(self):
                self.n = 0

            async def generate(self, prompt, max_new):
                self.n += 1
                return (TOOL, "stop") if "<tool>" not in prompt else ("<answer> A </answer>", "stop")

            async def search(self, args):
                res = [["T1\nx", "T2\ny", "T3\nz"]]
                return A.format_results(res), res

        row = {"qid": 4, "question": "q", "answer": ["a"], "data_source": "search_zero_2",
               "messages": [A.message("user", "Question: q")]}
        rec = asyncio.run(p1_run.run_question(row, 1, A.Renderer(FakeTokenizer(), [A.TOOL_SCHEMA]),
                                              FakeClients()))
        self.assertEqual(rec["k"], 5)
        self.assertEqual(len(rec["replays"]), 5)
        self.assertTrue(all(len(p) == 2 and p[0]["replayable"] for p in rec["replays"]))
        self.assertEqual(rec["replays"][0][0]["stats"]["permuted"], 1)
        st = S.question_stats(json.loads(json.dumps(rec)))
        self.assertEqual((st["k"], st["tot"], st["noise"]), (5, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
