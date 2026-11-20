import unittest

from work3.wiki_rsi_action_diagnosis import POLICIES, diagnose_action, summarize


def query_row(query, passages, correct):
    return {"query": query, "status": "executed" if query else "format", "passages": passages, "correct": correct}


class WikiActionDiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.case = {"question": "Where was Stone born?", "visible": [["book", "Written by John Stone"]],
                     "probe_title": "John Stone", "missing_facts": [["john stone", "born in london"]]}

    def test_reproduces_evidence_and_distinguishes_page_and_sentence_misses(self):
        classify = lambda action: diagnose_action(self.case, action)
        self.assertEqual(classify(query_row("John Stone", [("John Stone", "Born in London.")], 1)),
                         {"query": "John Stone", "outcome": "hit",
                          "literal_query_in_input": True, "exact_title_query": True})
        self.assertEqual(classify(query_row("Stone birthplace", [("John Stone", "He was a writer.")], 0))["outcome"],
                         "missing_sentence")
        self.assertEqual(classify(query_row("Stone", [("Stone", "Born in London.")], 0))["outcome"],
                         "missing_page")
        self.assertEqual(classify(query_row(None, [], 0))["outcome"], "not_executed")
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            classify(query_row("John Stone", [], 1))

    def test_pool_coverage_and_unissued_working_title_are_distinct(self):
        outcomes = [("hit", "missing_page"), ("missing_page", "hit"), ("missing_page", "missing_page")]
        rows = [{"known_title": True, "policies": {
            policy: {"outcome": pair[0 if policy == "frozen" else 1],
                     "literal_query_in_input": False, "exact_title_query": False} for policy in POLICIES}}
                for pair in outcomes]
        report = summarize(rows)
        self.assertEqual(report["any_policy_hits"], 2)
        self.assertEqual(report["any_policy_minus_best_single"], 1)
        self.assertEqual(report["frozen_only_hits"], 1)
        self.assertEqual(report["trained_only_hits"], 1)
        self.assertEqual(report["known_title_hit_no_policy_issued_title"], 1)
