import unittest

from work3.p0_retrieval_diag import SUPPORT_DATASETS
from work3.wiki_rsi_natural import MIN_GAP, MIN_PROBE, MIN_ROWS, score_case, summarize, visible_queries


class WikiNaturalGateTest(unittest.TestCase):
    def test_visible_query_favors_new_link_from_relevant_passage(self):
        question = "Which painter worked in the royal palace?"
        history = [("palace", "The royal palace housed Clara Jones, a painter."),
                   ("irrelevant", "The London Times has articles about queens.")]
        catalog = {"clara jones": {"clara jones"}, "london times": {"london times"}}
        rules, titles = visible_queries(question, history, catalog)
        self.assertEqual(rules["original"], question)
        self.assertEqual(rules["linked"], "clara jones")
        self.assertTrue(rules["linked_context"].startswith("clara jones "))
        self.assertNotIn("unobserved support page", titles)

    def test_oracle_and_visible_envelopes_require_one_complete_query(self):
        case = {"dataset": SUPPORT_DATASETS[0], "qid": "fixture",
                "queries": {"original": "question", "linked": "first", "linked_context": "context"},
                "visible_titles": ["first", "second"], "probe_title": "known", "strict_corpus_all": True,
                "missing_facts": [["page", "fact one"], ["page", "fact two"]]}
        results = {"question": [], "first": [("page", "fact one")], "context": [],
                   "second": [("page", "fact two")], "known": [("page", "fact one and fact two")]}
        row = score_case(case, results)
        self.assertTrue(row["known_title_beyond_visible"])
        self.assertFalse(row["any_visible_candidate"])
        self.assertFalse(row["any_fixed"])

    def test_pilot_gate_demands_headroom_in_both_datasets(self):
        def row(dataset, i, visible):
            return {"dataset": dataset, "original": False, "linked": False, "linked_context": False,
                    "strict_corpus_all": True, "known_title": i < MIN_PROBE, "any_fixed": False,
                    "any_visible_candidate": visible, "known_title_beyond_fixed": i < MIN_PROBE,
                    "visible_candidate_beyond_fixed": visible, "known_title_beyond_visible": i < MIN_PROBE and not visible}

        rows = [row(dataset, i, i < MIN_GAP) for dataset in SUPPORT_DATASETS for i in range(MIN_ROWS)]
        self.assertEqual(summarize(rows)[1], "measure_frozen_action_selection")
        rows = [row(dataset, i, False) for dataset in SUPPORT_DATASETS for i in range(MIN_ROWS)]
        self.assertEqual(summarize(rows)[1], "one_frozen_query_generation_check")
        self.assertEqual(summarize(rows[:MIN_ROWS])[1], "insufficient_headroom_for_training")
