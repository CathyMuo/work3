import unittest

from work3.wiki_rsi_repair_audit import analyze_people, birth_entries, candidates, choose_candidates
from work3.wiki_rsi_repair_pool import verified_birth


class WikiRepairAuditTest(unittest.TestCase):
    def test_normalization_accepts_same_title_full_name_and_date_spacing(self):
        title = "John Smith"
        docs = [(title, "Sir John William Smith (March 4 , 1965 &ndash; September 7 , 2018) was a director.")]
        observed = birth_entries(title, docs)
        self.assertEqual(observed["status"], "lead_name_mismatch")
        self.assertEqual(observed["strict"], verified_birth(title, docs))
        self.assertEqual(observed["normalized"][0], "1965")
        self.assertIsNone(birth_entries(title, [("Other Person", docs[0][1])])["normalized"])
        conflicted = docs + [(title, "John Smith (born May 4, 1970) was a director.")]
        self.assertIsNone(birth_entries(title, conflicted)["normalized"])

    def test_one_witness_foil_retains_named_page_and_excludes_answer_leakage(self):
        people = {"director": "John Smith", "writer": "Mary Jones"}
        returns = {"John Smith": [("John Smith", "John Smith (born 1965) was a director.")],
                   "Mary Jones": [("Mary Jones", "Mary Jones wrote several scripts.")]}
        evidence = {name: birth_entries(name, docs) for name, docs in returns.items()}
        call = lambda mode: candidates("Example Film", [], people, returns, evidence, mode)
        self.assertFalse(call("strict_paired")[1])
        stages, records = call("strict_one_witness")
        self.assertIn("source_with_clean_pair", stages)
        self.assertEqual([row["target"] for row in records], ["John Smith"])
        self.assertEqual(len(choose_candidates([records, records])), 1)
        returns["Mary Jones"].append(("History", "The year 1965 is mentioned."))
        self.assertFalse(call("strict_one_witness")[1])
        returns["Mary Jones"] = [("Another Page", "No dates")]
        evidence["Mary Jones"] = birth_entries("Mary Jones", returns["Mary Jones"])
        self.assertFalse(call("strict_one_witness")[1])

    def test_full_corpus_witness_separates_title_vs_chunk_ranking(self):
        corpus = {"john smith": [("John Smith", "John Smith (born 1965) was a director.")],
                  "mary jones": [("Mary Jones", "Mary Jones (born 1970) was a writer.")]}
        live = {"John Smith": [("Wrong Page", "Result one")], "Mary Jones": [("Mary Jones", "Later work.")]}
        _, counts, _ = analyze_people(live, corpus)
        self.assertEqual(counts["corpus_normalized_birth"], 2)
        self.assertEqual(counts["corpus_birth_but_title_missed"], 1)
        self.assertEqual(counts["corpus_birth_but_not_in_live_passages"], 1)
