import unittest

from work3.wiki_rsi_w2 import Retriever, check_anchors, measure, parse_query


class WikiW2Test(unittest.TestCase):
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
