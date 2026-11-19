import unittest

from work3.p0_retrieval_diag import SUPPORT_DATASETS
from work3.wiki_rsi_action_pilot import EVAL_MINS, MIN_EVAL_WITNESSES, MIN_TRAIN
from work3.wiki_rsi_action_pilot import literal_title, readiness, select_training


class WikiActionPilotTest(unittest.TestCase):
    def test_literal_stratum_uses_question_title_and_body(self):
        self.assertTrue(literal_title("Who arrived in Pérou?", [], "pérou"))
        self.assertTrue(literal_title("Who arrived?", [("River Town", "another clue")], "river town"))
        self.assertFalse(literal_title("Who arrived near the station?", [], "station house"))

    def test_training_requires_real_witness_nonleaking_foil_and_isolated_prefix(self):
        good = [("Example Film", "Example Film is a film directed by John Smith and written by Mary Jones.")]
        trace = {"source": "Example Film", "prefix": good,
                 "people": {"director": "John Smith", "writer": "Mary Jones"}}
        docs = {"John Smith": [("John Smith", "Sir John Edward Smith (born 5 March 1965) was a director.")],
                "Mary Jones": [("Mary Jones", "Mary Jones wrote several scripts.")]}
        rows = select_training([trace, trace], docs, set())
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["positive_query"], rows[0]["negative_query"]), ("John Smith", "Mary Jones"))
        leaking = {**trace, "prefix": [*good, ("Evaluation Page", "A fact.")]}
        self.assertFalse(select_training([leaking], docs, {"evaluation page"}))
        docs["Mary Jones"].append(("Other", "A 1965 article."))
        self.assertFalse(select_training([trace], docs, set()))

    def test_readiness_requires_both_strata_in_both_datasets(self):
        report = {ds: {label: {"n": n, "known_title": MIN_EVAL_WITNESSES}
                       for label, n in EVAL_MINS.items()} for ds in SUPPORT_DATASETS}
        train = [None] * MIN_TRAIN
        self.assertTrue(readiness(train, report))
        report[SUPPORT_DATASETS[0]]["not_literal"]["n"] -= 1
        self.assertFalse(readiness(train, report))
