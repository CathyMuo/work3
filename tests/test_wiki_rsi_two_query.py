import unittest

from work3.wiki_rsi_two_query import POLICIES, replay, summarize


class WikiTwoQueryTest(unittest.TestCase):
    def test_merge_passages_and_pay_only_unique_executed_queries(self):
        missing = [("target", "first fact"), ("target", "second fact")]
        first = {"query": "a", "passages": [("Target", "First fact.")]}
        second = {"query": "b", "passages": [("Target", "Second fact.")]}
        duplicate = {**second, "query": "a"}
        self.assertEqual(replay(first, second, missing), {"hit": True, "queries": 2, "duplicate": False})
        self.assertEqual(replay(first, duplicate, missing), {"hit": False, "queries": 1, "duplicate": True})
        self.assertEqual(replay({"query": None, "passages": []}, second, missing)["queries"], 1)

    def test_fixed_controls_are_compared_on_same_frozen_prefix(self):
        row = {policy: {"hit": policy.endswith("positive_only") or policy == "rule_original",
                        "queries": 2, "duplicate": False} for policy in POLICIES}
        report = summarize([row])
        self.assertEqual(report["rule_best_hits"], 1)
        self.assertEqual(report["policies"]["20261119-positive_only"]["net_vs_best_rule"], 0)
        self.assertEqual(report["policies"]["20261119-positive_only"]["net_vs_frozen"], 1)
