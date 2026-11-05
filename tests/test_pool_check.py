import unittest

from work3 import pool_check


def row(content, answer="Paris", source="search_zero_1"):
    return ([{"role": "user", "content": content}],
            {"ground_truth": {"target": [answer]}}, source)


SOLVER = "Answer the given question. You must conduct reasoning ... Question: x"
CHALLENGER = "\nYou are an expert in question generation. Craft one ..."


class PoolCheckTest(unittest.TestCase):
    def test_all_solver_prompts_pass(self):
        r = pool_check.check_rows([row(SOLVER), row(SOLVER, source="search_zero_2")])
        self.assertTrue(r["ok"])
        self.assertEqual(r["data_sources"], {"search_zero_1": 1, "search_zero_2": 1})

    def test_unreplaced_challenger_prompt_fails(self):
        r = pool_check.check_rows([row(SOLVER), row(CHALLENGER)])
        self.assertFalse(r["ok"])
        self.assertEqual(r["challenger_prompts"], 1)

    def test_empty_pool_fails(self):
        self.assertFalse(pool_check.check_rows([])["ok"])

    def test_empty_answers_are_counted(self):
        r = pool_check.check_rows([row(SOLVER, answer=""), row(SOLVER, answer=None)])
        self.assertEqual(r["empty_answers"], 2)
        self.assertTrue(r["ok"])

    def test_missing_target_counts_as_empty(self):
        r = pool_check.check_rows([([{"content": SOLVER}], {"ground_truth": {"target": None}}, "s")])
        self.assertEqual(r["empty_answers"], 1)


if __name__ == "__main__":
    unittest.main()
