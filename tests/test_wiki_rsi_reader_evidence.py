import unittest

from work3 import wiki_rsi_reader_evidence as read


class ReaderEvidenceTest(unittest.TestCase):
    def test_oracle_keeps_only_observed_support_from_its_original_page(self):
        docs = [("Hall", "The hall opened in Rio. It closed recently."), ("Museum", "The hall opened in Lima.")]
        found = read.observed_support(docs, {("hall", "the hall opened in rio"), ("hall", "the hall opened in lima"),
                                             ("unseen", "never in these passages")})
        self.assertEqual(found, [("Hall", "The hall opened in Rio. It closed recently")])
        self.assertEqual(read.excerpt("Writers discussed São Paulo &amp; Salvador.", "são paulo salvador", surrounding=0),
                         "São Paulo & Salvador")

    def test_compact_has_actual_windows_and_a_fixed_budget(self):
        docs = [(f"Place {i}", f"The hall in place {i} opened for visitors.") for i in range(11)]
        selected = read.compact_evidence("Where did the hall open?", "hall opening", docs)
        self.assertEqual(len(selected), read.MAX_WINDOWS)
        self.assertEqual(len({title for title, _ in selected}), read.MAX_WINDOWS)
        sources = dict(docs)
        self.assertTrue(all(body in sources[title] for title, body in selected))

    def test_complete_support_subset_is_decided_without_answer_correctness(self):
        rows = []
        for both_observed, predicted in [(True, False), (False, True)]:
            arms = {arm: {"all_reference_facts": both_observed, "correct": int(predicted),
                          "valid": True, "original_gap_complete": both_observed, "prompt_tokens": 10}
                    for arm in read.ARMS}
            rows.append({"dataset": "hotpotqa", "cached_correct": int(predicted), "arms": arms})
        result = read.summarize(rows)["hotpotqa"]
        self.assertEqual(result["all"]["n"], 2)
        self.assertEqual(result["complete_reference_in_both"]["n"], 1)
        self.assertEqual(result["complete_reference_in_both"]["arms"]["full"]["correct"], 0)
