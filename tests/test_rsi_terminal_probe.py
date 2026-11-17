import json
import unittest

from work3.rsi_failure_probe import linked_case
from work3.rsi_probe import Completion, decide
from work3.rsi_terminal_probe import (
    SEED, acquisition_replay, pool_numbers, pool_replay, run_pair, score_quantities,
)


class TerminalProbeTest(unittest.IsolatedAsyncioTestCase):
    def test_terminal_period_is_opt_in_and_does_not_relax_tool_calls(self):
        clean = Completion("Calculation. <answer>B</answer>.")
        self.assertEqual(decide(clean).status, "format_error")
        self.assertEqual(decide(clean, allow_answer_period=True).answer, "B")
        for raw in (
            "<answer>A</answer>... ",
            "<answer>A</answer> followed by more text.",
            "<answer>B</answer><answer>B</answer>.",
            '<tool_call>{"name":"search"}</tool_call><answer>B</answer>.',
        ):
            self.assertEqual(decide(Completion(raw), allow_answer_period=True).status, "format_error")
        self.assertEqual(decide(Completion(clean.text, "length"), allow_answer_period=True).status,
                         "truncated")

    async def test_empty_control_excludes_placebo_harm(self):
        case = linked_case("amended", 1, seed=SEED)
        placebo = linked_case("amended", 1, seed=SEED, placebo=True)
        correct_world = case.answers.index("B")
        history = [{"role": "user", "content": case.prompt}]
        seen = []

        async def generate(messages):
            self.assertEqual(messages[:-1], history)
            content = messages[-1]["content"]
            seen.append(content)
            label = "A" if placebo.changed_query in content else "B"
            return Completion(f"<answer>{label}</answer>.")

        result = await acquisition_replay(case, placebo, correct_world, history, [2], generate)
        self.assertEqual(len(seen), 3)
        self.assertTrue(result["all_valid"])
        self.assertFalse(result["relevant_only"])
        self.assertEqual({key: value["correct"] for key, value in result["arms"].items()},
                         {"relevant": True, "placebo": False, "empty": True})

    async def test_pool_logs_bad_sum_separately_from_bad_decision(self):
        case = linked_case("pool", 0, seed=SEED)
        world = case.answers.index("B")
        truth = pool_numbers(case, world)

        async def generate(history):
            context = history[-1]["content"]
            subtotal_provided = "provided: null." not in context
            values = truth if subtotal_provided else {**truth, "total": truth["total"] + 20}
            answer = "B" if subtotal_provided else "A"
            return Completion("<quantities>" + json.dumps(values) + f"</quantities><answer>{answer}</answer>.")

        result = await pool_replay(case, world, [], generate)
        self.assertEqual(result["arms"]["unassisted"]["numeric_status"],
                         "wrong_sum_given_correct_quantities")
        self.assertEqual(result["arms"]["verified_subtotal"]["numeric_status"],
                         "correct_values_correct_route")
        output = result["arms"]["verified_subtotal"]
        self.assertEqual(score_quantities({**output, "correct": False}, truth),
                         "correct_values_wrong_route")

    async def test_rejected_autonomous_completion_is_retained(self):
        case = linked_case("amended", 0, seed=SEED)
        placebo = linked_case("amended", 0, seed=SEED, placebo=True)

        async def generate(history):
            return Completion("No usable answer")

        row = await run_pair(case, placebo, generate)
        self.assertEqual(row["rejections"][0]["worlds"], (0, 1))
        self.assertEqual(row["rejections"][0]["completion"]["text"], "No usable answer")
        self.assertEqual([world["category"] for world in row["worlds"]], ["format_error"] * 2)
