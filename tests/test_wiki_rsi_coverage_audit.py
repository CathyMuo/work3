import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from work3.wiki_rsi_coverage_audit import case_coverage, scan_corpus, score_probes


class WikiCoverageAuditTest(unittest.TestCase):
    def test_same_title_boundary_and_other_title_do_not_change_old_coverage(self):
        expected = {("shared page", "a sentence across the boundary"): False,
                    ("shared page", "an intact sentence"): True,
                    ("shared page", "an alias witness"): False,
                    ("second page", "an alias witness"): False,
                    ("second page", "a line that never occurs"): False}
        rows = ["Shared Page\nThe text has a sentence across", "Shared Page\nthe boundary. An intact sentence.",
                "Second Page\nAn alias witness.", "Third Page\nAn alias witness."]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wiki.jsonl"
            path.write_text("".join(json.dumps({"contents": row}) + "\n" for row in rows), encoding="utf-8")
            # The expected exact flag for the second page must agree with the actual corpus.
            with self.assertRaisesRegex(ValueError, "did not reproduce"):
                scan_corpus(path, expected, Mock())
            expected["second page", "an alias witness"] = True
            _, count, boundary, other = scan_corpus(path, expected, Mock())
        self.assertEqual(count, 4)
        self.assertEqual(boundary, {("shared page", "a sentence across the boundary"): [1, 2]})
        self.assertEqual(other, {("shared page", "an alias witness"): {"title": "second page", "row": 3}})
        case = {"dataset": "hotpotqa", "qid": "case", "probe_title": "shared page",
                "corpus": [{"title": title, "sentence": sentence, "exact_match": exact}
                           for (title, sentence), exact in expected.items() if title == "shared page"]}
        result = case_coverage(case, boundary, other)
        self.assertFalse(result["strict_all"])
        self.assertFalse(result["strict_or_boundary_all"])
        self.assertFalse(result["strict_or_other_title_all"])
        self.assertTrue(result["strict_or_alternative_all"])

    def test_oracle_union_can_add_coverage_without_single_query_gain(self):
        case = {"dataset": "hotpotqa", "qid": "case", "missing_pages": 2, "strict_all": True,
                "probe_title": "page a", "facts": [{"title": "page a", "sentence": "evidence a"},
                                                   {"title": "page b", "sentence": "evidence b"}]}
        returned = {"page a": [("Page A", "Evidence A.")],
                    "page b": [("Page B", "Evidence B.")]}
        result = score_probes(case, returned, [["Page A", "Evidence A."]])
        self.assertTrue(result["w3b_primary_exact_ordered"])
        self.assertEqual(result["logical_queries"], 2)
        self.assertFalse(result["best_single_all"])
        self.assertTrue(result["union_completes_beyond_single"])
        self.assertFalse(result["union_all_outside_strict_corpus"])
