import unittest

from work3.wiki_rsi_repair_pool import film_credits, repair_case, verified_birth, visible_credits


class WikiRepairPoolTest(unittest.TestCase):
    source = "Example Film"
    text = "Example Film is a film directed by John Smith and written by Mary Jones."
    answers = {
        "John Smith": [("John Smith", "John Smith (born 5 March 1965) was a director.")],
        "Mary Jones": [("Mary Jones", "Mary Jones (June 9, 1970 – 2017) was a writer.")],
    }

    def test_accept_only_distinct_unambiguous_visible_credits(self):
        self.assertEqual(film_credits(self.source, self.text), {"director": "John Smith", "writer": "Mary Jones"})
        coauthored = self.text.replace("John Smith and written", "John Smith and Amy Cole, written")
        self.assertEqual(film_credits(self.source, coauthored), {"writer": "Mary Jones"})
        comma = self.text.replace("John Smith and written", "John Smith, Amy Cole, written")
        self.assertEqual(film_credits(self.source, comma), {"writer": "Mary Jones"})
        self.assertEqual(visible_credits(self.source, [("Other Film", self.text)], set()), ({}, None))
        self.assertEqual(visible_credits(self.source, [(self.source, self.text)], {"mary jones"}), ({}, None))

    def test_birth_verifier_requires_same_page_lead_and_unique_year(self):
        self.assertEqual(verified_birth("John Smith", self.answers["John Smith"])[0], "1965")
        self.assertIsNone(verified_birth("John Smith", [("Other Page", self.answers["John Smith"][0][1])]))
        inconsistent = [*self.answers["John Smith"], ("John Smith", "John Smith (born 1975) was a director.")]
        self.assertIsNone(verified_birth("John Smith", inconsistent))

    def test_repair_case_requires_no_answer_in_history_or_matching_foil(self):
        prefix = [(self.source, self.text)]
        people, bridge = visible_credits(self.source, prefix, set())
        case = repair_case(self.source, prefix, bridge, people, self.answers, set())
        self.assertNotEqual(case["target"]["query"], case["foil"]["query"])
        self.assertNotEqual(case["answer"], case["foil"]["year"])
        self.assertIsNone(repair_case(self.source, prefix, bridge, people, self.answers, {"john smith"}))
        history = prefix + [("A date list", "1965, 1970")]
        self.assertIsNone(repair_case(self.source, history, bridge, people, self.answers, set()))
