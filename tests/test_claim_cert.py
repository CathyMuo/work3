import math
import random
import unittest

from work3.claim_cert_stats import (ALPHAS, CONFIGS, COST, GRID, TARGETS, aggregate, auc, binom_cdf, decide,
                                    escalated, evaluate, final_scores, fit_rule, judge_calibration, judge_errors,
                                    ltt, run_split, split_of)


def brute_cdf(f, m, p):
    return sum(math.comb(m, j) * p ** j * (1 - p) ** (m - j) for j in range(f + 1))


def synthetic_claims(n=3000, seed=1, true_rate=0.2):
    """Scores grow more informative with more passages; stage-2 configs separate true from false best."""
    generator = random.Random(seed)
    strength = {"q1": 0.6, "q3": 1.0, "q5": 1.2, "q10": 1.5, "q20": 1.8, "qa3": 1.1, "qa10": 1.6, "qa20": 1.9}
    claims = []
    for index in range(n):
        truth = generator.random() < true_rate
        stratum = "known" if truth else generator.choice(["wrong", "unknown", "partial"])
        base = generator.gauss(0, 1)
        scores = {}
        for key in CONFIGS:
            value = strength[key] * (1.5 if truth else -1.5) + base + generator.gauss(0, 1)
            scores[key] = 1 / (1 + math.exp(-value))
        claims.append({"qid": f"c{index}", "truth": truth, "stratum": stratum, "scores": scores,
                       "self_judge": scores["q20"] >= 0.5})
    return claims


class Thresholds(unittest.TestCase):
    def test_binomial_tail_matches_brute_force(self):
        for f, m, p in [(0, 10, 0.1), (3, 20, 0.1), (7, 50, 0.2), (49, 50, 0.05), (2, 5, 0.5)]:
            self.assertAlmostEqual(binom_cdf(f, m, p), brute_cdf(f, m, p), places=10)
        self.assertEqual(binom_cdf(5, 5, 0.1), 1.0)
        self.assertGreater(binom_cdf(10, 20000, 0.1), 0.0 - 1e-300)

    def test_grid_is_fixed_and_increasing(self):
        self.assertEqual(len(GRID), 50)
        self.assertTrue(all(a < b for a, b in zip(GRID, GRID[1:])))
        self.assertAlmostEqual(GRID[0], 1 / (1 + math.exp(4)))

    def test_ltt_certifies_only_clean_regions(self):
        scores = [0.999] * 200 + [0.3] * 200
        truth = [True] * 200 + [False] * 200
        threshold = ltt(scores, truth, 0.1)
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, 0.3)
        self.assertEqual(evaluate(scores, truth, threshold), {"tpr": 1.0, "kept": 0.5, "fdr": 0.0})
        self.assertIsNone(ltt([0.9] * 30, [False] * 30, 0.1))
        self.assertEqual(evaluate([0.9], [True], None)["kept"], 0.0)

    def test_ltt_controls_fdr_on_random_data(self):
        violations = 0
        for seed in range(40):
            claims = synthetic_claims(2000, seed)
            calib, test = claims[:1000], claims[1000:]
            threshold = ltt([c["scores"]["q3"] for c in calib], [c["truth"] for c in calib], 0.1)
            result = evaluate([c["scores"]["q3"] for c in test], [c["truth"] for c in test], threshold)
            violations += result["fdr"] > 0.15
        self.assertLessEqual(violations, 4)


class Policies(unittest.TestCase):
    def test_escalation_fraction_and_below_rule(self):
        generator = random.Random(3)
        stage1 = [generator.random() for _ in range(1000)]
        truth = [score > 0.7 for score in stage1]
        fitted = fit_rule(stage1, truth, 0.1, 0.3, "near")
        share = sum(escalated(score, fitted) for score in stage1) / len(stage1)
        self.assertAlmostEqual(share, 0.3, delta=0.02)
        below = fit_rule(stage1, truth, 0.1, 0.9, "below")
        centre = below["centre"]
        flagged = [score for score in stage1 if escalated(score, below)]
        self.assertTrue(all(math.log(s / (1 - s)) < centre for s in flagged))
        stage2 = [1.0 - score for score in stage1]
        scores, flags = final_scores(stage1, stage2, fitted)
        self.assertEqual(sum(flags), round(share * 1000))
        self.assertTrue(all(score == (s2 if flag else s1) for score, s1, s2, flag in zip(scores, stage1, stage2, flags)))

    def test_split_fractions(self):
        counts = {"tune": 0, "calib": 0, "test": 0}
        for index in range(20000):
            counts[split_of(f"q{index}", 0)] += 1
        self.assertAlmostEqual(counts["tune"] / 20000, 0.25, delta=0.02)
        self.assertAlmostEqual(counts["test"] / 20000, 0.5, delta=0.02)
        self.assertNotEqual(split_of("q1", 0), split_of("q1", 1)) if split_of("q1", 0) == "tune" else None

    def test_run_split_shapes_and_cost_matching(self):
        claims = synthetic_claims(4000, 7)
        result = run_split(claims, 0)
        for alpha in ALPHAS:
            uniform = result["uniform"][str(alpha)]
            self.assertEqual(set(uniform), set(CONFIGS))
            for target in TARGETS:
                cell = result["adaptive"][str(alpha)][str(target)]
                self.assertLessEqual(cell["cost"], target + 0.6)
                affordable = [row["tpr"] for row in uniform.values() if row["cost"] <= cell["cost"] + 1e-9]
                self.assertAlmostEqual(cell["gain"], cell["tpr"] - max(affordable), places=9)
        judged = run_split(claims, 0, "self_judge")
        self.assertIn("0.1", judged["uniform"])

    def test_aggregate_and_decision(self):
        claims = synthetic_claims(3000, 11)
        splits = [run_split(claims, repeat) for repeat in range(3)]
        rows = aggregate(splits)
        self.assertEqual(len(rows), len(ALPHAS) * len(TARGETS))
        self_judge = {"errors": judge_errors(claims, "self_judge"),
                      "calibration": judge_calibration([run_split(claims, r, "self_judge") for r in range(2)])}
        self.assertIn("fp_gap_wrong_unknown", self_judge["errors"])
        validity = {"claims": True}
        verdict = decide(rows, self_judge, validity)["verdict"]
        self.assertIn(verdict, {"GO", "PIVOT", "NO-GO"})
        self.assertEqual(decide(rows, self_judge, {"claims": False})["verdict"], "INCOMPLETE")
        go_row = [{"alpha": 0.1, "target": 10, "mean_gain": 0.12, "adaptive_fdr": 0.08}]
        no_judge = {"calibration": {"0.1": {"uniform_q3_ratio": 1.0}}, "errors": {"fp_gap_wrong_unknown": 0.0}}
        self.assertEqual(decide(go_row, no_judge, validity)["verdict"], "GO")
        broken = {"calibration": {"0.1": {"uniform_q3_ratio": 1.6}}, "errors": {"fp_gap_wrong_unknown": 0.12}}
        low = [{"alpha": 0.1, "target": 10, "mean_gain": 0.02, "adaptive_fdr": 0.08}]
        self.assertEqual(decide(low, broken, validity)["verdict"], "PIVOT")
        self.assertEqual(decide(low, no_judge, validity)["verdict"], "NO-GO")
        unsafe = [{"alpha": 0.1, "target": 10, "mean_gain": 0.2, "adaptive_fdr": 0.14}]
        self.assertEqual(decide(unsafe, no_judge, validity)["verdict"], "NO-GO")

    def test_auc(self):
        self.assertEqual(auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), 1.0)
        self.assertEqual(auc([0.5, 0.5], [False, True]), 0.5)
        self.assertIsNone(auc([0.1, 0.2], [True, True]))

    def test_costs_cover_configs(self):
        self.assertEqual(set(COST), set(CONFIGS))


class Pipeline(unittest.TestCase):
    def test_claims_and_analysis_without_gpus(self):
        import tempfile
        from pathlib import Path

        import work3.claim_cert as cc

        questions = [{"qid": f"q{i}", "question": f"Question {i}?", "prop": "occupation"} for i in range(600)]
        knownness = {f"q{i}": {"answer": "" if i == 5 else f"answer {i}", "abstains": i == 6,
                               "greedy_correct": i % 5 == 0, "stratum": "known" if i % 5 == 0 else "wrong"}
                     for i in range(600)}
        claims = cc.build_claims(questions, knownness, "run")
        self.assertEqual(len(claims), 598)
        self.assertEqual(len(cc.build_claims(questions, knownness, "smoke")), cc.SMOKE_CLAIMS)
        generator = random.Random(5)
        verified = {}
        for claim in claims:
            sign = 1.5 if claim["truth"] else -1.5
            verified[claim["qid"]] = {"scores": {key: 1 / (1 + math.exp(-(sign + generator.gauss(0, 1))))
                                                 for key in CONFIGS}}
        with tempfile.TemporaryDirectory() as folder:
            records, result = cc.analysis({"directory": Path(folder)}, claims, verified, None, "smoke")
        self.assertEqual(len(records), 598)
        self.assertEqual(result["splits"], 3)
        self.assertGreater(result["verifier"]["q20"]["auc"], 0.8)
        self.assertIsNone(result["external_judge"])


if __name__ == "__main__":
    unittest.main()
