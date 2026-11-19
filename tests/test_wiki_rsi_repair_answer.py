import unittest

from work3.wiki_rsi_repair_answer import ARMS, mask_year, normalized_docs, score, summarize_group


class WikiRepairAnswerTest(unittest.TestCase):
    def test_redacts_year_in_every_return_including_encoded_and_full_width(self):
        docs = normalized_docs([
            ("Title １９６５", "Facts in &#49;965 and 1965. Keep 1966 and X1965."),
            ("Another title", "The span 1965–1967 and catalog_1965."),
        ])
        self.assertEqual(mask_year(docs, "1965"), [
            ["Title [year omitted]", "Facts in [year omitted] and [year omitted]. Keep 1966 and X1965."],
            ["Another title", "The span [year omitted]–1967 and catalog_[year omitted]."],
        ])

    def test_explicit_abstention_is_valid_and_malformed_year_scores_zero(self):
        output = {"text": "<answer>unknown</answer>", "finish": "stop", "tokens": 5}
        self.assertEqual(score(output, "1965")["abstain"], 1)
        self.assertTrue(score(output, "1965")["valid"])
        self.assertEqual(score({**output, "text": "Born in <answer>1965</answer>."}, "1965")["correct"], 0)
        self.assertEqual(score({**output, "text": "<answer>1965</answer>", "finish": "length"}, "1965")["correct"], 0)
        self.assertEqual(score({**output, "text": "<answer>1965</answer>"}, "1965")["correct"], 1)

    def test_pairing_keeps_invalid_generation_in_primary_denominator(self):
        rows = [{"arms": {arm: {"valid": arm != "target_year_masked", "correct": int(arm == "target"),
                               "abstain": 0} for arm in ARMS}}]
        contrast = summarize_group(rows)["target_pairs"]["target_year_masked"]
        self.assertEqual(contrast["strict_em"]["wins"], 1)
        self.assertEqual(contrast["strict_em"]["n"], 1)
        self.assertEqual(contrast["both_valid"]["n"], 0)
