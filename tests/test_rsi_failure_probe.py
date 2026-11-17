import json
import unittest

from work3.rsi_failure_probe import FAMILIES, linked_case, run_batch, run_evidence
from work3.rsi_probe import Completion, evaluate
from work3.rsi_tasks import make_case


def call(queries):
    payload = {"name": "search", "arguments": {"query_list": list(queries)}}
    return Completion("<tool_call>" + json.dumps(payload) + "</tool_call>")


def answer(label):
    return Completion(f"<answer>{label}</answer>")


class FollowupTest(unittest.IsolatedAsyncioTestCase):
    def test_new_families_and_independent_controls(self):
        for family in FAMILIES:
            for i in range(30):
                case, filler = linked_case(family, i), linked_case(family, i, placebo=True)
                first, second = case.worlds
                self.assertEqual([k for k in first if first[k] != second[k]], [case.changed_query])
                self.assertFalse(set(first) & set(filler.worlds[0]))
                for world, label in zip(case.worlds, case.answers):
                    order = world[case.oracle[0]]
                    if family == "amended":
                        current = world["order " + order["current_order"]]
                        stock = world["stock " + current["sku"]]["on_hand"]
                        need = current["quantity"]
                    else:
                        stock = sum(world["stock " + order[lot]]["on_hand"]
                                    for lot in ("lot_a", "lot_b"))
                        need = order["quantity"]
                    self.assertEqual(label, "A" if stock >= need else "B")
                    self.assertEqual(set(case.answers), {"A", "B"})

    async def test_batch_accepts_same_action_and_counts_individual_lookups(self):
        case = make_case("bundle", 20261117, 0)

        async def generate(history):
            if len(history) == 2:
                return call(case.oracle[:1])
            if len(history) == 4:
                return call(case.oracle[1:])
            results = json.loads(history[-1]["content"])["results"]
            self.assertEqual([item["query"] for item in results], list(case.oracle[1:]))
            world_index = int(results[0]["record"] == case.worlds[1][case.changed_query])
            return answer(case.answers[world_index])

        result = await run_batch(case, "replay", generate)
        strict, batch = (result["arms"][arm] for arm in ("single", "batch"))
        self.assertEqual(strict["rejections"][0]["reason"], "multiple_queries")
        self.assertEqual(strict["rejections"][0]["executed_queries"], 1)
        self.assertEqual(batch["correct"], [True, True])
        self.assertEqual(batch["fork_at_query"], 3)
        self.assertEqual(batch["cost"]["tool_executions"], 6)
        self.assertEqual(result["unique_generation_calls"], 4)

    async def test_batch_does_not_execute_part_of_an_over_budget_action(self):
        case = make_case("bundle", 20261117, 0)

        async def generate(history):
            if len(history) == 2:
                return call(case.oracle[:1])
            return call(case.oracle)

        row = await evaluate(case, generate, 3, allow_batch=True, run_oracle=False)
        self.assertIsNone(row["fork_at_query"])
        self.assertEqual(row["costs"]["autonomous"]["tool_executions"], 2)
        self.assertEqual([b["status"] for b in row["autonomous"]], ["budget_exhausted"] * 2)

    async def test_evidence_interventions_share_identical_observed_history(self):
        case, placebo = linked_case("amended", 0), linked_case("amended", 0, placebo=True)
        prompts = []

        async def generate(history):
            if len(history) == 2:
                return call(case.oracle[:1])
            if len(history) == 4:
                return answer("A")
            prompts.append(history)
            content = history[-1]["content"]
            return answer("B" if case.changed_query in content else "A")

        row = await run_evidence(case, placebo, generate)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0][:-1], prompts[1][:-1])
        by_label = {world["label"]: world for world in row["worlds"]}
        self.assertEqual(by_label["B"]["category"], "rescued_only_by_relevant_evidence")
        self.assertEqual(by_label["B"]["missing"], list(case.oracle[1:]))
        contents = by_label["B"]["intervention"]
        self.assertNotIn(placebo.oracle[0], contents["placebo"]["message"]["content"])
        self.assertIn(placebo.oracle[-1], contents["placebo"]["message"]["content"])
        self.assertEqual(by_label["A"]["category"], "autonomous_correct")
