import unittest
from unittest.mock import patch

from work3 import wiki_rsi_reader_bridge as bridge


class ReaderBridgeTest(unittest.TestCase):
    def test_matching_token_caps_isolates_pairwise_link_preference(self):
        docs = [("Archive", "A letter is kept at Golden Library."),
                ("Museum", "Opening date documented in local media."),
                ("Golden Library", "It opened in 1922.")]
        follow_up = {docs[-1]}
        with patch.object(bridge, "INPUT_CAP", 20), patch.object(
                bridge.live, "answer_prompt", side_effect=lambda _tok, _q, docs: [1] * (10 + 5 * len(docs))):
            independent = bridge.pack(None, "What is the date?", "", docs, follow_up, "independent")
            linked = bridge.pack(None, "What is the date?", "", docs, follow_up, "linked")
        self.assertIn(docs[1][0], [title for title, _ in independent[1]])
        self.assertIn(docs[0][0], [title for title, _ in linked[1]])
        self.assertIn(docs[2][0], [title for title, _ in linked[1]])
        self.assertEqual(len(independent[0]), len(linked[0]))

    def test_bridge_decision_requires_joint_retention_and_both_answer_comparisons(self):
        subset = {"arms": {"linked": {"all_reference_facts": 21}, "independent": {"all_reference_facts": 15}}}
        by_control = {"full": {"wins": 7, "losses": 2}, "independent": {"wins": 6, "losses": 1}}
        groups = {"2wikimultihopqa": {"all": {"linked_vs": by_control}, "full_has_all_reference_facts": subset}}
        self.assertEqual(bridge.verdict(groups), "eligible_for_unseen_2wiki_replication")
        by_control["full"]["losses"] = 3
        self.assertEqual(bridge.verdict(groups), "no_go_for_linked_packing")
