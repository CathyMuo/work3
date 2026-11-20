import unittest

import torch

from work3.wiki_rsi_evidence_switch import losses, verified_example


class EvidenceSwitchTest(unittest.TestCase):
    def setUp(self):
        self.question = "Which of the two landmarks opened earlier?"
        self.first = {"query": "Town library opening", "witness": ("Town library", "It opened in 1921."),
                      "observation": [("Town library", "It opened in 1921.")],
                      "returned": [("Town library", "It opened in 1921.")]}
        self.second = {"query": "Hill museum opening", "witness": ("Hill museum", "It opened in 1919."),
                       "observation": [("Hill museum", "It opened in 1919.")],
                       "returned": [("Hill museum", "It opened in 1919.")]}

    def test_same_queries_have_opposite_preferences_for_different_seen_evidence(self):
        row = verified_example(self.question, [], [self.first, self.second])
        self.assertEqual(row["gains"], [[0, 1], [1, 0]])
        self.assertEqual(row["question"], self.question)
        self.assertEqual(row["queries"], [self.first["query"], self.second["query"]])
        self.assertNotIn("witness", row)

    def test_rejects_shared_evidence_leak_and_returns_without_complementary_facts(self):
        self.assertIsNone(verified_example(self.question, self.first["observation"], [self.first, self.second]))
        missing = {**self.second, "returned": [("Hill museum", "The opening was widely reported.")]}
        self.assertIsNone(verified_example(self.question, [], [self.first, missing]))

    def test_interaction_cancels_a_context_independent_query_preference(self):
        scores = torch.tensor([[0.0, 1.0], [2.0, 0.0]], requires_grad=True)
        baseline, combined = losses(scores)
        bias = torch.tensor([3.0, -4.0])
        biased_baseline, biased_combined = losses(scores + bias)
        torch.testing.assert_close(combined - baseline, biased_combined - biased_baseline)
        combined.backward()
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertLess(scores.grad[1, 0].item(), 0)
