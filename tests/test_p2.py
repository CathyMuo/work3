import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from work3 import ncp_reward as R, p1_agent as A, p2_gate as G
from test_p1_agent import FakeTokenizer, TOOL


class DifficultyTest(unittest.TestCase):
    def test_drzero_matches_upstream(self):
        self.assertEqual([R.drzero_difficulty(k, 5) for k in range(6)], [0, 1.0, 0.75, 0.5, 0.25, 0])

    def test_beta_shape(self):
        vals = [R.beta_difficulty(k, 7) for k in range(8)]
        self.assertEqual((vals[0], vals[7]), (0.0, 0.0))
        self.assertAlmostEqual(vals[1], 1.0)
        self.assertTrue(all(a > b for a, b in zip(vals[1:6], vals[2:7])))

    def test_signal_share(self):
        scores = [1, 0, 0, 0, 0]
        self.assertEqual(R.signal_share(scores, [(1, 1)] + [(0, 0)] * 4), (1.0, 0.2, 0.0))
        w, tot, noise = R.signal_share(scores, [(1, 0)] + [(0, 0)] * 4)
        self.assertAlmostEqual(noise, 0.1); self.assertAlmostEqual(w, 0.5)
        self.assertEqual(R.signal_share(scores, [(1, 0)] * 3 + [(0, 0)] * 2)[0], 0.0)
        self.assertEqual(R.signal_share([1] * 5, [(1, 0)] * 5)[0], 0.0)

    def test_arm_reward(self):
        s = [1, 0, 0, 0, 0]
        self.assertEqual(R.arm_reward("drzero", s)["difficulty"], 1.0)
        self.assertAlmostEqual(R.arm_reward("ncp", s, [(1, 0)] + [(0, 0)] * 4)["difficulty"], 0.5)
        with self.assertRaises(ValueError):
            R.arm_reward("other", s)


class FakeClients:
    """Answers correctly only when passages arrive in the original order."""

    def __init__(self):
        self.calls = {"generate": 0}

    async def generate(self, prompt, max_new):
        self.calls["generate"] += 1
        if "<tool>" not in prompt:
            return TOOL, "stop"
        ok = "Doc 1 (Title: T1)" in prompt and "Doc 2 (Title: T2)" in prompt   # original order only
        return ("<answer> yes </answer>" if ok else "<answer> no </answer>"), "stop"

    async def search(self, args):
        res = [["T1\na", "T2\nb", "T3\nc"]]
        return A.format_results(res), res


class ScoreQuestionTest(unittest.TestCase):
    def renderer(self):
        return A.Renderer(FakeTokenizer(), [A.TOOL_SCHEMA])

    def test_ncp_detects_order_dependence(self):
        msgs = [A.message("user", "Question: q")]
        out = asyncio.run(R.score_question("ncp", msgs, "yes", self.renderer(), FakeClients(), [1, 2, 3]))
        # originals all correct (k = 5) -> zero difficulty; every replay (permuted) fails
        self.assertEqual((out["k"], out["difficulty"], out["unreplayable"]), (5, 0.0, 0))
        self.assertAlmostEqual(out["s2_noise"], 0.0)            # both replays fail -> pairs agree

    def test_ncp_weight_from_replay_disagreement(self):
        pairs = [(1, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        out = R.arm_reward("ncp", [1, 0, 0, 0, 0], pairs)
        self.assertAlmostEqual(out["w"], 0.5)
        self.assertAlmostEqual(out["difficulty"], 0.5)

    def test_ncp_rollout_counts(self):
        cl = FakeClients()
        asyncio.run(R.score_question("ncp", [A.message("user", "q")], "yes", self.renderer(), cl, [0]))
        self.assertEqual(cl.calls["generate"], 5 * 2 + 10)      # 5 two-turn originals + 10 one-turn replays

    def test_beta_uses_seven_rollouts_and_no_replays(self):
        cl = FakeClients()
        out = asyncio.run(R.score_question("beta", [A.message("user", "q")], "yes", self.renderer(), cl, [0]))
        self.assertEqual((out["n"], cl.calls["generate"]), (7, 14))

    def test_score_all_closes_and_reports(self):
        res, calls = asyncio.run(R.score_all("drzero", ["a?", "b?"], ["yes", "no"], self.renderer(),
                                             FakeClients, 1, "Question: {question}"))
        self.assertEqual([r["k"] for r in res], [5, 0])
        self.assertEqual(calls["generate"], 20)


def p0_summary(a2, a3, verdict="PASS"):
    em = {d: a3 for d in ("nq", "triviaqa")}
    return {"verdict": verdict, "iterations": {"1": {"average": a2 - 0.01}, "2": {"average": a2},
                                               "3": {"average": a3, "em": em}}}


class GateTest(unittest.TestCase):
    def test_go(self):
        r = G.decide(p0_summary(0.32, 0.326), p0_summary(0.33, 0.335), p0_summary(0.34, 0.350))
        self.assertEqual(r["verdict"], "GO")
        self.assertAlmostEqual(r["primary"]["ncp_minus_drzero"], 0.024)

    def test_margins_and_decline(self):
        dz, beta = p0_summary(0.32, 0.326), p0_summary(0.33, 0.335)
        self.assertEqual(G.decide(dz, beta, p0_summary(0.34, 0.345))["verdict"], "NO-GO")   # +1.0 over beta ok, +1.9 over dz
        self.assertEqual(G.decide(dz, p0_summary(0.33, 0.342), p0_summary(0.34, 0.350))["verdict"], "NO-GO")
        self.assertEqual(G.decide(dz, beta, p0_summary(0.36, 0.350))["verdict"], "NO-GO")   # declined
        self.assertEqual(G.decide(dz, beta, p0_summary(0.346, 0.346))["verdict"], "GO")     # exact margins

    def test_invalid(self):
        good = p0_summary(0.32, 0.33)
        self.assertEqual(G.decide(p0_summary(0.3, 0.3, "FAIL"), good, good)["verdict"], "INVALID")
        self.assertEqual(G.decide(good, {"verdict": "INVALID"}, good)["verdict"], "INVALID")

    def test_secondary_eta_and_compute(self):
        p1 = {"drzero": {"pools": {"3": {"eta_mixed": 0.6}}}, "ncp": {"pools": {"3": {"eta_mixed": 0.45}}}}
        logs = {"beta": {"completion_tokens": 100}, "ncp": {"completion_tokens": 160}}
        r = G.decide(p0_summary(0.32, 0.326), p0_summary(0.33, 0.335), p0_summary(0.34, 0.35), p1, logs)
        self.assertTrue(r["secondary"]["ncp_eta_drop_ge_0.10"])
        self.assertFalse(r["secondary"]["compute_matched"])

    def test_reward_usage_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "r.jsonl"
            log.write_text(json.dumps({"calls": {"generate": 3, "completion_tokens": 50}, "seconds": 2}) + "\n"
                           + json.dumps({"calls": {"generate": 1, "completion_tokens": 5}, "seconds": 1}) + "\n")
            self.assertEqual(G.reward_usage(str(log)), {"reward_calls": 2, "generate_requests": 4,
                                                         "completion_tokens": 55, "reward_seconds": 3.0})
            paths = {}
            for name, s in (("dz", p0_summary(0.32, 0.326)), ("b", p0_summary(0.33, 0.335)),
                            ("n", p0_summary(0.34, 0.35))):
                paths[name] = Path(d) / f"{name}.json"; paths[name].write_text(json.dumps(s))
            out = Path(d) / "p2.json"
            code = G.main(["--drzero", str(paths["dz"]), "--beta", str(paths["b"]), "--ncp", str(paths["n"]),
                           "--ncp-reward-log", str(log), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["verdict"], "GO")


if __name__ == "__main__":
    unittest.main()
