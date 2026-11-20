import unittest
from unittest.mock import patch

from work3 import wiki_rsi_switch_data as data
from work3 import wiki_rsi_switch_train as train


class WikiEvidenceSwitchPipelineTest(unittest.TestCase):
    def test_pair_keeps_both_titles_and_flips_real_missing_evidence(self):
        sources = []
        for name, year in (("Hill Library", 1924), ("Town Museum", 1933)):
            returned = [(name, f"{name} was opened in {year}. It occupies a large brick building.")]
            source = data.observed_fact(name, year, "opening", returned)
            self.assertIsNotNone(source)
            self.assertIsNone(data.observed_fact(name, year - 1, "opening", returned))
            sources.append({**source, "query": f"{name} opened date"})
        row = data.verified_pair(*sources, "opening")
        self.assertEqual(row["gains"], [[0, 1], [1, 0]])
        self.assertEqual([name for name, _ in row["states"][0]], [name for name, _ in row["states"][1]])
        self.assertIn("1924", " ".join(body for _, body in row["states"][0]))
        self.assertNotIn("1933", " ".join(body for _, body in row["states"][0]))
        self.assertEqual(row["aliases"], ["Hill Library" if "earlier" in row["question"] else "Town Museum"])

    def test_reader_validity_and_action_scores(self):
        example = {"question": "Which opened first?", "states": [[], []], "returns": [[], []], "aliases": ["Hill Library"]}
        texts = ["<answer>unknown</answer>", "<answer>Hill Library</answer>", "<answer>Hill Library</answer>", "bad"]
        with patch.object(train.reader, "pack", return_value=([1], [])):
            matrix, valid = train.reader_matrix(example, None, None, "cpu",
                lambda *_: [{"text": text, "finish": "stop"} for text in texts])
        self.assertTrue(train.gated(matrix))
        self.assertFalse(valid)
        self.assertEqual(train.measure_choices([1, 0], matrix), {"both_correct": 1, "covered": 2, "em": 2})
        self.assertEqual(train.measure_choices([0, 0], matrix), {"both_correct": 0, "covered": 1, "em": 1})
