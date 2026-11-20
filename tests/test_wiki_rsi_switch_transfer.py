import unittest
from unittest.mock import patch

import torch

from work3 import wiki_rsi_action_model as trainer
from work3 import wiki_rsi_switch_transfer as transfer


class SwitchTransferTest(unittest.TestCase):
    def test_year_rule_needs_no_reader_labels(self):
        row = {"pages": ["Library", "Gallery"], "states": [
            [["Gallery", "It is a venue."], ["Library", "It opened in 1900."]],
            [["Gallery", "It opened in 1914."], ["Library", "It is a venue."]],
        ]}
        self.assertEqual(transfer.date_rule(row), [1, 0])

    def test_duplicate_queries_and_empty_pool_do_not_use_hindsight(self):
        class Tokenizer:
            eos_token_id = 2

            def __call__(self, query, **kwargs):
                return {"input_ids": [10 + len(query)]}

        def outcome(text=None, value=0):
            return {"second_query": text, "correct": value, "evidence_complete": value, "queries": 2 if text else 1}

        case = {"dataset": "hotpotqa", "qid": "x", "title_stratum": "literal",
                "visible": [["Seed", "Original clue"]], "question": "What is the clue?"}
        result = {"dataset": "hotpotqa", "qid": "x", "title_stratum": "literal", "first_query": "first",
                  "first_passages": [["New", "Observed query result"]],
                  "arms": {"first_only": outcome(), "frozen_follow_up": outcome("same"),
                           "linked_context": outcome("same", 1), transfer.live.POSITIVE[0]: outcome("new", 1),
                           transfer.live.POSITIVE[1]: outcome("first")}}
        with patch.object(transfer.live, "pack", return_value=([4], [])) as pack:
            row, = transfer.natural_examples(Tokenizer(), [case], [result])
        seen = pack.call_args.args[2]
        self.assertEqual(seen, case["visible"] + result["first_passages"])
        self.assertEqual([value["correct"] for value in row["outcomes"]], [0, 1])
        class Model:
            def eval(self):
                pass
        with patch.object(trainer, "mean_logprobs", return_value=torch.tensor([0.0, 1.0])):
            choices = transfer.natural_policy(Model(), [row, {**row, "completions": []}], 0, "cpu")
        self.assertEqual([value["correct"] for value in choices], [1, 0])
