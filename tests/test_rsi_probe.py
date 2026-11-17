import json
import unittest

from work3.rsi_probe import (ORACLE_FINISH, Completion, decide, evaluate, metrics,
                             oracle_call, tool_message)
from work3.rsi_tasks import FAMILIES, SPLITS, make_case, make_cases


def call(query: str) -> Completion:
    return Completion("<tool_call>" + json.dumps({
        "name": "search", "arguments": {"query_list": [query]},
    }) + "</tool_call>")


def answer(label: str) -> Completion:
    return Completion(f"<answer>{label}</answer>")


class TasksTest(unittest.TestCase):
    def test_exactly_one_record_changes_and_targets_flip(self):
        for family in FAMILIES:
            for index in range(20):
                with self.subTest(family=family, index=index):
                    case = make_case(family, 20261117, index)
                    a, b = case.worlds
                    self.assertEqual(a.keys(), b.keys())
                    self.assertEqual({k for k in a if a[k] != b[k]}, {case.changed_query})
                    changed = case.changed_query
                    self.assertEqual({key for key in a[changed] if a[changed][key] != b[changed][key]},
                                     {"on_hand"})
                    self.assertEqual(set(case.answers), {"A", "B"})
                    self.assertEqual(case, make_case(family, 20261117, index))
                    control = make_case(family, 20261117, index, True)
                    self.assertEqual(len(set(control.answers)), 1)
                    self.assertEqual(control.oracle, ())

    def test_split_and_counts(self):
        self.assertTrue(set(SPLITS["development"]).isdisjoint(SPLITS["heldout"]))
        cases = make_cases("heldout", 4, 2, 9)
        self.assertEqual(sum(c.control for c in cases), 4)
        self.assertEqual(len(cases), len({c.id for c in cases}))
        self.assertEqual({c.family for c in cases}, set(SPLITS["heldout"]))

    def test_bundle_short_circuit(self):
        case = make_case("bundle", 42, 1)
        failed, passed = case.answers.index("B"), case.answers.index("A")
        queries = (case.oracle[0], case.changed_query)
        self.assertTrue(case.sufficient(failed, queries))
        self.assertFalse(case.sufficient(passed, queries))


class ParsingTest(unittest.TestCase):
    def test_invalid_action_is_separate_from_an_answer(self):
        for completion in (
            Completion("<answer>A</answer><answer>B</answer>"),
            Completion("<tool_call>"),
            Completion("<tool_call>{}</tool_call>"),
            call("stock wrong"),
            Completion(
                call("stock SKU-123456").text + "<answer>A</answer>"),
        ):
            self.assertEqual(decide(completion).status, "format_error")
        self.assertEqual(decide(Completion("<answer>A</answer>", "length")).status, "truncated")
        self.assertEqual(decide(call("stock SKU-123456")).query, "stock SKU-123456")
        self.assertEqual(decide(answer("B")).answer, "B")


class ProbeTest(unittest.IsolatedAsyncioTestCase):
    async def test_shared_prefix_is_evaluated_once_and_oracle_replays_raw_returns(self):
        case = make_case("bundle", 42, 0)
        order, target, second = case.oracle
        seen = []

        async def generate(history):
            seen.append(history)
            if history[-1]["content"] == ORACLE_FINISH:
                self.assertEqual(history[-1]["role"], "user")
                history = history[:-1]
            depth = (len(history) - 2) // 2
            if depth < len(case.oracle):
                return call(case.oracle[depth])
            response = history[5]
            world = int(response == tool_message(case.worlds[1], target))
            self.assertEqual(history[2::2], [oracle_call(q) for q in (order, target, second)])
            self.assertEqual(history[3::2], [tool_message(case.worlds[world], q)
                                            for q in (order, target, second)])
            return answer(case.answers[world])

        row = await evaluate(case, generate, 4)
        self.assertEqual(row["diagnosis"], "autonomous_solved")
        self.assertEqual(row["fork_at_query"], 2)
        self.assertEqual(row["costs"]["autonomous"]["calls"], 6)
        self.assertEqual(row["costs"]["autonomous"]["tool_executions"], 6)
        self.assertEqual(len(seen), 8)
        self.assertEqual(sum(h[-1]["content"] == ORACLE_FINISH for h in seen), 2)
        self.assertEqual(row["oracle_correct"], [True, True])

    async def test_early_guesses_vs_observation_errors(self):
        case = make_case("single", 42, 0)

        async def oracle_recovers(history):
            if len(history) == 2:
                return answer("A")
            i = int(history[-2] == tool_message(case.worlds[1], case.changed_query))
            return answer(case.answers[i])

        async def cannot_use(history):
            if len(history) == 2:
                return call(case.changed_query)
            return answer("A")

        gap = await evaluate(case, oracle_recovers, 4)
        self.assertEqual(gap["diagnosis"], "decisive_evidence_not_acquired")
        self.assertEqual(gap["costs"]["autonomous"]["calls"], 1)
        used = await evaluate(case, cannot_use, 4)
        self.assertEqual(used["diagnosis"], "post_observation_error")
        self.assertTrue(used["full_evidence_both"])
        self.assertEqual(metrics([gap, used])["oracle"]["pairs_correct"], 1)

    async def test_controls_and_query_caps(self):
        control = make_case("order", 42, 0, True)

        async def respond(history):
            return answer(control.answers[0])

        async def retry(history):
            return call("order ORD-000000")

        solved = await evaluate(control, respond, 4)
        self.assertEqual(solved["oracle"], [])
        self.assertEqual(metrics([solved])["controls"]["pairs_with_tool"], 0)
        capped = await evaluate(control, retry, 3)
        self.assertEqual(capped["autonomous"][0]["status"], "budget_exhausted")
        self.assertEqual(capped["costs"]["autonomous"]["calls"], 4)
        self.assertEqual(capped["costs"]["autonomous"]["tool_executions"], 6)


if __name__ == "__main__":
    unittest.main()
