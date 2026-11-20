import unittest
from unittest.mock import patch

from work3 import wiki_rsi_two_query_live as live


class TwoQueryLiveTest(unittest.TestCase):
    def test_dedup_preserves_slot_and_single_query_budget(self):
        class Retriever:
            def search(self, queries):
                self.queries = queries
                return [[("Page", query)] for query in queries]

        retriever = Retriever()
        rows, physical = live.search_batch(retriever, ["already", "new", "new"], ["already", "a", "b"])
        self.assertEqual(retriever.queries, ["new"])
        self.assertEqual(physical, 1)
        self.assertEqual(rows, [(None, []), ("new", [("Page", "new")]), ("new", [("Page", "new")])])

    def test_every_contextual_follow_up_sees_the_actual_shared_first_results(self):
        case = {"dataset": "hotpotqa", "qid": "a", "question": "Who opened it?", "title_stratum": "literal",
                "visible": [("Base", "Earlier evidence")], "missing_facts": [("target", "Rory opened it")],
                "aliases": ["Rory"], "linked_context": "rule"}

        class Retriever:
            def search(self, queries):
                passages = {"first": [("Clue", "Actual first result")],
                            "learned": [("Target", "Rory opened it.")]}
                return [passages.get(query, []) for query in queries]

        def generate(model, tok, prompts, device, max_new):
            out = []
            for (question, docs), in prompts:
                if max_new == live.ANSWER_TOKENS:
                    self.assertIsNone(change.call_args.args[2] if len(change.call_args.args) > 2 else None)
                    text = "<answer>Rory</answer>" if ("Target", "Rory opened it.") in docs else "<answer>unknown</answer>"
                elif ("Clue", "Actual first result") not in docs:
                    self.assertEqual(docs, [("Base", "Earlier evidence")])
                    text = "first"
                else:
                    self.assertEqual(question, case["question"])
                    text = "learned" if len(change.call_args.args) > 2 and change.call_args.args[2] else "follow"
                out.append({"text": text, "finish": "stop"})
            return out

        progress = type("Progress", (), {"update": lambda self, *args, **kwargs: None})()
        with patch.object(live, "set_adapters") as change, patch.object(live, "pack",
                side_effect=lambda tok, question, docs, render, max_new: ([(question, docs)], docs)):
            rows, counts = live.evaluate([case], None, None, [], dict.fromkeys(live.POSITIVE, [1]),
                                         Retriever(), "cpu", progress, None, generate)
        self.assertEqual(counts["shared_first"], 1)
        self.assertEqual(rows[0]["first_passages"], [("Clue", "Actual first result")])
        self.assertEqual(rows[0]["arms"]["frozen_follow_up"]["correct"], 0)
        for name in live.POSITIVE:
            self.assertEqual(rows[0]["arms"][name]["correct"], 1)
            self.assertEqual(rows[0]["arms"][name]["queries"], 2)
