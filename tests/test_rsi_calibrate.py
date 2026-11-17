import json
import unittest
from collections import Counter

from work3.rsi_calibrate import (
    COMPARE, CONDITIONS, CONFIRMATION_SEED, DIAGNOSTIC_SEED, diagnostic_cases, oracle_prompt,
    run_case, summarize,
)
from work3.rsi_probe import ORACLE_FINISH, Completion, SYSTEM, message, tool_message
from work3.rsi_tasks import make_case


class CalibrationCasesTest(unittest.TestCase):
    def test_manifest_fixes_cohorts_and_control_counts(self):
        cases = diagnostic_cases()
        counts = Counter((case.family, case.control) for case, _ in cases)
        self.assertEqual(counts, {
            ("bundle", False): 12, ("reserved", False): 12,
            ("bundle", True): 4, ("reserved", True): 4,
        })
        self.assertEqual(len({case.id for case, _ in cases}), 32)
        for case, group in cases:
            if group.startswith("raw_stock_"):
                world = case.worlds[case.answers.index("B")]
                enough = world[case.changed_query]["on_hand"] >= world[case.oracle[0]]["quantity"]
                self.assertEqual(enough, group == "raw_stock_sufficient")
            self.assertNotEqual(case.worlds, make_case(
                case.family, CONFIRMATION_SEED, int(case.id.split("-")[1]), case.control).worlds)

    def test_evidence_is_identical_and_in_same_order_in_all_conditions(self):
        case = make_case("bundle", DIAGNOSTIC_SEED, 1)
        world = case.worlds[0]
        expected_returns = [tool_message(world, query)["content"] for query in case.oracle]
        direct = oracle_prompt(case, world, "tool_direct")
        compare = oracle_prompt(case, world, "tool_compare")
        self.assertEqual(direct[:2], [message("system", SYSTEM), message("user", case.prompt)])
        self.assertEqual([msg["content"] for msg in direct if msg["role"] == "tool"], expected_returns)
        self.assertEqual([msg["tool_calls"][0]["function"]["arguments"]["query_list"][0]
                          for msg in direct if msg["role"] == "assistant"], list(case.oracle))
        self.assertEqual(compare[:-1], direct[:-1])
        self.assertEqual(direct[-1]["content"], ORACLE_FINISH)
        self.assertEqual(compare[-1]["content"], COMPARE)
        for variant, instruction in (("inline_direct", ORACLE_FINISH), ("inline_compare", COMPARE)):
            inline = oracle_prompt(case, world, variant)
            self.assertEqual(inline[:2], direct[:2])
            self.assertEqual(inline[-1]["role"], "user")
            sections = inline[-1]["content"].split("\n\n")
            self.assertEqual(sections[1:-1], [key + "\n" + payload for key, payload
                                           in zip(case.oracle, expected_returns)])
            self.assertEqual(sections[-1], instruction)


class CalibrationRunTest(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_reused_and_rejected_batch_preserved(self):
        case = make_case("bundle", DIAGNOSTIC_SEED, 0)
        seen = []
        invalid = '<tool_call>' + json.dumps({
            "name": "search", "arguments": {"query_list": list(case.oracle[1:])},
        }) + '</tool_call>'

        async def generate(history):
            seen.append(history)
            if len(history) == 2:
                payload = {"name": "search", "arguments": {"query_list": list(case.oracle[:1])}}
                return Completion("<tool_call>" + json.dumps(payload) + "</tool_call>")
            if history[-1]["role"] == "tool":
                return Completion(invalid, "stop", 325, 56)
            return Completion("<answer>A</answer>", "stop", 350, 7)

        row = await run_case(case, "prior_format_error", generate, None)
        self.assertEqual(set(row["conditions"]), set(CONDITIONS))
        for output in row["conditions"].values():
            self.assertEqual([x["status"] for x in output], ["answered", "answered"])
        self.assertEqual(len(seen), 10)
        self.assertEqual(seen[2], oracle_prompt(case, case.worlds[0], "tool_direct"))
        self.assertEqual(seen[3], oracle_prompt(case, case.worlds[1], "tool_direct"))
        events = row["autonomous"]["rejections"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["worlds"], (0, 1))
        self.assertEqual(events[0]["executed_queries"], 1)
        self.assertEqual(events[0]["reason"], "multiple_queries")
        self.assertEqual(events[0]["completion"]["text"], invalid)
        self.assertEqual(events[0]["history"][-1], tool_message(case.worlds[0], case.oracle[0]))
        self.assertEqual([b["status"] for b in row["autonomous"]["branches"]],
                         ["format_error", "format_error"])

    async def test_confirmation_only_runs_chosen_arm_and_enforces_both_gates(self):
        histories = []

        async def generate(history):
            histories.append(history)
            return Completion("<answer>A</answer>")

        rows = []
        for family in ("bundle", "reserved"):
            case = make_case(family, CONFIRMATION_SEED, 0)
            rows.append(await run_case(case, "fresh_confirmation", generate, "inline_compare"))
        self.assertTrue(all(set(row["conditions"]) == {"inline_compare"} for row in rows))
        self.assertTrue(all("autonomous" not in row for row in rows))
        self.assertEqual(len(histories), 4)
        result = summarize(rows, "inline_compare")
        self.assertFalse(result["confirmation"]["pass"])
        for family in ("bundle", "reserved"):
            self.assertEqual(result["confirmation"]["by_family"][family]["observed"], {
                "valid_world_rate": 1.0, "pair_accuracy": 0.0, "label_b_accuracy": 0.0,
            })
