import json
import tempfile
import unittest
from pathlib import Path

from test_wiki_rsi_audit import record
from work3.evidence_ref import answer_keys, question_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS
from work3.wiki_rsi_candidates import action_list, best_of_pool, build_pool, visible_catalog


class WikiCandidateTest(unittest.TestCase):
    def test_candidate_queries_are_visible_and_keep_original_fallback(self):
        catalog = {"river town": {"river town"}, "hidden page": {"hidden page"},
                   "coastal trail": {"coastal trail"}}
        choices = visible_catalog("Which trail is there?",
                                  [("already read", "the coastal trail reached river town")], catalog)
        self.assertEqual(choices, ["coastal trail", "river town", "already read"])
        self.assertEqual(action_list("Which trail is there?", choices),
                         [*choices, "Which trail is there?"])
        self.assertEqual(action_list("Which trail is there?", []), ["Which trail is there?"])

    def test_eval_pool_excludes_previously_seen_support_pages(self):
        tests = {ds: {} for ds in SUPPORT_DATASETS}
        refs = {ds: {} for ds in SUPPORT_DATASETS}
        for question, title in (("overlap", "old page"), ("eligible", "new page")):
            key = question_key(question)
            tests["hotpotqa"][key] = ["Answer"]
            refs["hotpotqa"][key] = {"answers": list(answer_keys(["Answer"])),
                                     "titles": {"known page": ["first fact"], title: ["second fact"]}}
        rows = [record(name, [("Known page", "First fact.")]) for name in ("overlap", "eligible")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dump.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pool, stats = build_pool(path, tests, refs, {"old page"})
        self.assertEqual(stats["hotpotqa"]["overlaps_diagnostic_pages"], 1)
        self.assertEqual([case[0] for case in pool["hotpotqa"]], [question_key("eligible")])

    def test_candidate_ceiling_does_not_union_partial_retrievals(self):
        self.assertEqual(best_of_pool([{"any_missing_fact": True, "all_missing_facts": False}] * 2),
                         {"any_missing_fact": True, "all_missing_facts": False})
