import math
import unittest

from work3.budget_rlvr_metrics import BASELINES, METHOD, PREFIX, QUESTIONS, REPEATS, compare, exact_panel


def panel(short_cost=25, ended=False):
    cases = [{"id": str(i)} for i in range(QUESTIONS)]
    rows = []
    prefix = [5] * (2 if ended else PREFIX)
    for i, case in enumerate(cases):
        for repeat in range(REPEATS):
            correct = (i + repeat) % 4 == 0
            right = (i + repeat) % 5 == 0
            rows.append({"qid": case["id"], "repeat": repeat, "prompt_tokens": 100,
                         "prefix_ids": prefix, "prefix_ended": ended, "prefix_correct": correct if ended else False,
                         "prefix_has_box": ended, "base_tokens": 100 + len(prefix),
                         "short_ids": [] if ended else [4] * 8, "long_ids": [] if ended else [4] * 100,
                         "short_ended": True, "long_ended": True,
                         "short_correct": correct if ended else right, "long_correct": correct,
                         "short_tokens": 0 if ended else short_cost, "long_tokens": 0 if ended else 100})
    return rows, cases


class BudgetMetricsTests(unittest.TestCase):
    def test_same_cost_uniform_variance_and_all_comparators(self):
        rows, cases = panel()
        report = compare(rows, cases)
        self.assertTrue(report["checks"]["matched_cost"])
        self.assertEqual(set(report["arms"]), {METHOD, *BASELINES})
        self.assertAlmostEqual(report["arms"]["uniform"]["rmse"], math.sqrt(len(rows) / 4) / len(rows))
        self.assertAlmostEqual(report["arms"][METHOD]["expected_tokens"], 534 * len(rows))
        self.assertEqual(report["short_wrong_long_right"], report["late_rescue_count"])
        self.assertGreater(report["short_right_long_wrong"], 0)

    def test_unaffordable_short_probe_cannot_pass(self):
        rows, cases = panel(short_cost=70)
        report = compare(rows, cases)
        self.assertEqual(report["verdict"], "NO_GO")
        self.assertIn(METHOD, report["infeasible"])
        self.assertFalse(report["checks"]["matched_cost"])

    def test_no_censored_samples_and_missing_rows(self):
        rows, cases = panel(ended=True)
        self.assertEqual(compare(rows, cases)["reason"], "no_truncated_prefixes_in_fold")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            exact_panel(rows[:-1], cases, REPEATS)
