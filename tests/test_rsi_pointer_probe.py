import json
import unittest

from work3.rsi_pointer_probe import FAMILY_COUNTS, pointer_case, probe_case
from work3.rsi_probe import Completion, decide, evaluate


def lookup(query):
    return Completion("<tool_call>" + json.dumps({"name": "search", "arguments": {
        "query_list": [query],
    }}) + "</tool_call>")


class PointerProbeTest(unittest.IsolatedAsyncioTestCase):
    def test_only_the_genuine_link_selects_the_current_order(self):
        for family in FAMILY_COUNTS:
            case, archived = pointer_case(family, 4)
            root, current, stock = case.oracle
            a, b = case.worlds
            self.assertEqual([key for key in a if a[key] != b[key]], [stock])
            self.assertEqual(set(case.answers), {"A", "B"})
            self.assertNotEqual(a[current]["sku"], a[archived]["sku"])
            if family == "status_list":
                item = next(v for v in a[root]["versions"] if v["status"] == "approved")
            elif family == "revision_list":
                item = max(a[root]["versions"], key=lambda v: v["revision"])
            else:
                item = {"order_id": a[root]["current_order"]}
            self.assertEqual("order " + item["order_id"], current)

    def test_exact_bare_label_requires_explicit_opt_in(self):
        self.assertEqual(decide(Completion("B")).status, "format_error")
        self.assertEqual(decide(Completion("B"), allow_bare_answer=True).answer, "B")
        for bad in ("A or B", "A.", "Route B", "<answer>A</answer><answer>B</answer>"):
            self.assertEqual(decide(Completion(bad), allow_bare_answer=True).status, "format_error")
        self.assertEqual(decide(Completion("B", "length"), allow_bare_answer=True).status,
                         "truncated")

    async def test_forced_actions_share_first_lookup_and_charge_budget(self):
        case, archived = pointer_case("status_list", 0)
        root, current, stock = case.oracle

        async def generate(history):
            queries = [call["function"]["arguments"]["query_list"][0]
                       for msg in history for call in msg["tool_calls"] or []]
            self.assertEqual(queries[0], root)
            if current in queries and stock not in queries:
                return lookup(stock)
            if stock in queries:
                value = json.loads(history[-1]["content"])["record"]
                return Completion(case.answers[int(value == case.worlds[1][stock])])
            return Completion("B")

        row = await probe_case(case, archived, generate)
        a = row["arms"]
        self.assertEqual(a["force_current"]["prefix_queries"], (root, current))
        self.assertEqual(a["force_archived"]["prefix_queries"], (root, archived))
        self.assertTrue(a["force_current"]["pair_correct"])
        self.assertFalse(a["force_archived"]["pair_correct"])
        self.assertTrue(a["force_current"]["decisive_stock_in_both_worlds"])
        self.assertEqual(a["force_current"]["cost"]["tool_executions"], 6)
        self.assertEqual(a["force_current"]["fork_at_query"], 3)

    async def test_seed_prefix_cannot_expose_world_identity(self):
        case, _ = pointer_case("revision_list", 3)

        async def impossible(_):
            self.fail("Generation must not start with a divergent preset")

        with self.assertRaises(ValueError):
            await evaluate(case, impossible, 4, seed_queries=(case.changed_query,), run_oracle=False)
