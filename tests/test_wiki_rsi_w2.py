import unittest
from unittest.mock import patch

from work3.wiki_rsi_w2 import Retriever, check_anchors, measure, parse_query
from work3.wiki_rsi_w2_contract import compatible_query, title_cue
from work3.wiki_rsi_w2_budget import matching_title
from work3.wiki_rsi_w2_answer import ranked_prefix, summarize as summarize_answers


class WikiW2Test(unittest.TestCase):
    def test_real_retrieval_keeps_rank_order_under_context_budget(self):
        docs = [("First", "irrelevant"), ("Second", "too long"), ("Third", "missing fact")]
        with patch("work3.wiki_rsi_w2_answer.w1.with_passages",
                   side_effect=lambda tok, prefix, articles: [1] * (2000 * len(articles))):
            result, prompt = ranked_prefix(None, [], docs, [1])
        self.assertEqual(result, docs[:1])
        self.assertEqual(len(prompt), 2000)

    def test_answer_summary_keeps_unmatched_cases_in_other_comparisons(self):
        def arm(value):
            return {"correct": value, "valid": True, "p0_em": value}

        row = {"dataset": "hotpotqa", "w1_rescued": True,
               "coverage": {"title_query": {"all_missing_facts": True}},
               "arms": {"empty": arm(0), "original_query": arm(0), "title_query": arm(1)}}
        results = summarize_answers([row])["hotpotqa"]
        self.assertEqual(results["title_vs_empty"]["strict_em"]["wins"], 1)
        self.assertEqual(results["title_vs_matched_irrelevant"]["strict_em"]["n"], 0)

    def test_budget_title_control_excludes_visible_support_and_answer(self):
        lengths = {name: 3 for name in ("target page", "given town", "support page",
                                         "lakeside paris", "old forest")}
        lengths["much longer candidate title"] = 6
        result = matching_title("qid", "target page", lengths, {"support page"}, lengths,
                                "Who left this town?", [("location", "moved to given town")], ["Paris"])
        self.assertEqual(result, ["old forest"])

    def test_compatible_parser_recovers_only_one_explicit_search(self):
        call = '<tool_call>{"name":"search","arguments":{"query_list":["river town"]}}</tool_call>'
        self.assertEqual(compatible_query("<think>Need to look up the town.</think>" + call, "stop"),
                         ("river town", "native_search"))
        self.assertEqual(compatible_query("<think>Need to look up the town.</think><query>river town</query>", "stop"),
                         ("river town", "query_after_think"))
        self.assertEqual(compatible_query(call + call, "stop")[0], None)
        self.assertIsNone(compatible_query(call.replace('["river town"]', '["river", "town"]'), "stop")[0])
        self.assertIsNone(compatible_query(call.replace('"search"', '"answer"'), "stop")[0])
        self.assertIsNone(compatible_query(call + " extra", "stop")[0])
        self.assertIsNone(compatible_query(call, "length")[0])

    def test_literal_cue_comes_from_original_question_or_passages(self):
        self.assertEqual(title_cue("Who worked there?", [("source", "moved to river town")], "river town"),
                         "returned_passage")
        self.assertEqual(title_cue("Who left River Town?", [], "river town"), "question")
        self.assertEqual(title_cue("Which plant grew there?", [("other", "we moved to riverside")], "river"),
                         "no_literal_cue")

    def test_query_parser_accepts_one_stopped_query(self):
        self.assertEqual(parse_query(" <query>River history</query>\n", "stop"), "River history")
        self.assertIsNone(parse_query("<query>River\nhistory</query>", "stop"))
        self.assertIsNone(parse_query("<query>   </query>", "stop"))
        self.assertIsNone(parse_query("<query>River history</query>", "length"))
        self.assertIsNone(parse_query("<query>River</query><query>City</query>", "stop"))

    def test_retrieval_coverage_checks_title_and_exact_sentence(self):
        required = [["page a", "first fact"], ["page b", "second fact"]]
        result = measure([("Page A", "This is the first fact."), ("Other", "Second fact.")], required)
        self.assertEqual(result, {"matched_facts": 1, "any_missing_fact": True, "all_missing_facts": False})
        result = measure([("Page A", "First fact."), ("Page B", "Second fact.")], required)
        self.assertTrue(result["all_missing_facts"])

    def test_anchors_compare_top_results_in_order_per_dataset(self):
        class StubRetriever:
            def search(self, queries):
                return [[("B", "second"), ("A", "first")] if number < 2
                        else [("A", "first"), ("B", "second")] for number, _ in enumerate(queries)]

        anchors = [{"dataset": ds, "query": f"{ds} query {number}",
                    "expected": [["a", "first"], ["b", "second"]]}
                   for ds in ("hotpotqa", "2wikimultihopqa") for number in range(4)]
        passed, stats = check_anchors(StubRetriever(), anchors)
        self.assertFalse(passed)
        self.assertEqual(stats["hotpotqa"]["exact_ordered"], 2)
        self.assertEqual(stats["2wikimultihopqa"]["exact_ordered"], 4)
        with self.assertRaises(ValueError):
            Retriever("https://en.wikipedia.org/retrieve")
