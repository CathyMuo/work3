import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from work3 import wiki_rsi_candidate_audit as audit
from work3.wiki_rsi_candidates import scan_corpus


class WikiCandidateAuditTest(unittest.TestCase):
    def test_corpus_counts_title_even_when_its_chunk_lacks_exact_sentence(self):
        articles = [
            {"contents": '"River Town"\nA different sentence.'},
            {"contents": '"River Town"\nThe missing fact.'},
            {"contents": '"Other Page"\nAn absent fact.'},
        ]
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder) / "wiki.jsonl"
            corpus.write_text("".join(json.dumps(row) + "\n" for row in articles), encoding="utf-8")
            titles, _, rows, title_rows, found = scan_corpus(
                corpus, [{"question": "Who left River Town?", "visible": []}], Mock(),
                missing_by_title={"river town": {"the missing fact", "an absent fact"}})
        self.assertEqual(rows, 3)
        self.assertEqual(titles["river town"], {"river town"})
        self.assertEqual(title_rows["river town"], 2)
        self.assertEqual(found, {("river town", "the missing fact")})

    def test_question_rerank_uses_only_visible_titles_and_preserves_ties(self):
        titles = ["river school", "west county", "east county", "river town"]
        self.assertEqual(audit.question_first("Which county contains River Town?", titles),
                         ["river town", "river school", "west county", "east county"])

    def test_probe_separates_chunk_choice_from_title_and_never_unions_queries(self):
        question = "Who went there?"
        missing = [["town", "town detail"], ["school", "school detail"]]
        case = {"dataset": "hotpotqa", "qid": "test", "question": question, "probe_title": "town",
                "ranks": {"legacy": 5, "question_rank": 1}, "target_in_catalog": True,
                "missing_facts": missing, "lists": {"legacy": ["school", "town"], "question_rank": ["town"]},
                "corpus": [{"title": title, "title_passages": 2, "exact_match": True} for title, _ in missing]}
        returned = {
            question: [],
            "town": [("Town", "wrong chunk")],
            f"town. {question}": [("Town", "town detail")],
            "school": [("School", "school detail")],
            f"school. {question}": [("School", "school detail")],
        }
        old = {"original_question": {"passages": []}, "title_probe": {"passages": returned["town"]}}
        result = audit.score_case(case, returned, old)
        self.assertTrue(result["corpus_all"])
        self.assertTrue(result["probes"]["title"]["target_page_in_top3"])
        self.assertFalse(result["probes"]["title"]["any_target_fact"])
        self.assertTrue(result["probes"]["title_question"]["all_target_facts"])
        self.assertFalse(result["arms"]["legacy_title_question_best3"]["all_missing_facts"])
        groups = audit.summarize([result])["hotpotqa"]
        self.assertEqual(groups["one_missing_page"]["n"], 0)
        self.assertEqual(groups["multiple_missing_pages"]["probe_title_page_without_target_fact"], 1)
        self.assertEqual(groups["all_states"]["probe_title_question_target_fact_but_title_after_legacy_top3"], 1)
