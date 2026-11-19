import unittest
from unittest.mock import patch

from work3.p0_retrieval_diag import SUPPORT_DATASETS
from work3.wiki_rsi_natural import MIN_ROWS
from work3.wiki_rsi_natural_query import CONTROLS, MIN_NET_WINS, parse_query, prompt_ids, summarize


class WikiNaturalQueryTest(unittest.TestCase):
    def test_only_one_query_string_can_be_executed(self):
        self.assertEqual(parse_query('Query: "Plymouth railway opening"', "stop"), ("Plymouth railway opening", "executed"))
        self.assertEqual(parse_query("<query>County Palace history</query>", "stop"), ("County Palace history", "executed"))
        self.assertEqual(parse_query("First query\nSecond query", "stop"), (None, "multiline"))
        self.assertEqual(parse_query("First query\u2028Second query", "stop"), (None, "multiline"))
        self.assertEqual(parse_query("some text", "length"), (None, "length"))
        self.assertEqual(parse_query("<answer>1890</answer>", "stop"), (None, "invalid"))

    def test_prompt_uses_question_and_retrieved_material_only(self):
        with patch("work3.wiki_rsi_natural_query.w1.render_ids", return_value=[1]) as render:
            self.assertEqual(prompt_ids(None, "What happened in the county?", [("County", "Some history.")]), [1])
        messages = render.call_args.args[1]
        self.assertIn("Some history.", messages[1]["content"])
        self.assertNotIn("gold", messages[1]["content"].lower())
        self.assertEqual(render.call_args.kwargs["tools"], ())

    def test_net_gain_gate_retains_unexecuted_cases(self):
        def row(ds, i):
            correct = i < MIN_NET_WINS
            return {"dataset": ds, "status": "executed" if correct else "context_overflow",
                    "any_missing_fact": correct, "all_missing_facts": correct, "probe_title_literal": False,
                    "baseline": dict.fromkeys(CONTROLS, False)}

        rows = [row(ds, i) for ds in SUPPORT_DATASETS for i in range(MIN_ROWS)]
        summary, decision = summarize(rows)
        self.assertEqual(decision, "qualifies_for_independent_action_training_pilot")
        self.assertEqual(summary[SUPPORT_DATASETS[0]]["pairwise_all_missing"]["any_fixed"]["n"], MIN_ROWS)
        rows[-1]["baseline"]["any_fixed"] = True
        self.assertEqual(summarize(rows)[1], "close_current_training_route")
