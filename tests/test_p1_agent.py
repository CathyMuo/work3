import asyncio
import json
import os
import unittest

import numpy as np

from work3 import p1_agent as A

QWEN_TOK = os.environ.get("WORK3_QWEN_TOKENIZER", "/mnt/wh/RAG/models/Qwen/Qwen2.5-0.5B-Instruct")


class FakeTokenizer:
    """Deterministic stand-in: one token per 4 characters."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        parts = []
        for m in messages:
            tc = json.dumps(m.get("tool_calls")) if m.get("tool_calls") else ""
            parts.append(f"<{m['role']}>{m['content']}{tc}")
        return "".join(parts) + ("<gen>" if add_generation_prompt else "")

    def __call__(self, text):
        return {"input_ids": [list(range(len(text[0]) // 4 + 1))]}


def scripted(texts):
    """Generator returning queued (text, finish) pairs; records prompts."""
    queue = list(texts)
    prompts = []

    async def gen(prompt, max_new):
        prompts.append((prompt, max_new))
        item = queue.pop(0)
        return item if isinstance(item, tuple) else (item, "stop")
    gen.prompts = prompts
    return gen


def fake_search(passages=None):
    calls = []

    async def search(args):
        calls.append(args)
        res = passages or [["T1\nA", "T2\nB", "T3\nC"] for _ in args["query_list"]]
        return A.format_results(res), res
    search.calls = calls
    return search


TOOL = '<think>need</think>\n<tool_call>\n{"name": "search", "arguments": {"query_list": ["q1"]}}\n</tool_call>'
USER = [A.message("user", "Answer ... Question: who?")]


def run(coro):
    return asyncio.run(coro)


class ParseTest(unittest.TestCase):
    def test_valid_call_normalises_content(self):
        normed, calls, err = A.parse_tool_calls(TOOL)
        self.assertEqual(normed, "<think>need</think>")
        self.assertEqual(calls, [{"id": "0", "type": "function",
                                  "function": {"name": "search", "arguments": {"query_list": ["q1"]}}}])
        self.assertIsNone(err)

    def test_bad_json_keeps_raw_content(self):
        text = "x <tool_call>{not json}</tool_call>"
        normed, calls, err = A.parse_tool_calls(text)
        self.assertEqual((normed, calls, err), (text, [], "json"))

    def test_unknown_tool_and_non_dict_arguments_are_dropped(self):
        t1 = '<tool_call>{"name": "browse", "arguments": {}}</tool_call>'
        t2 = '<tool_call>{"name": "search", "arguments": "[1]"}</tool_call>'
        self.assertEqual(A.parse_tool_calls(t1)[1], [])
        self.assertEqual(A.parse_tool_calls(t2)[1], [])
        self.assertEqual(A.parse_tool_calls(t1)[0], t1)

    def test_list_payload_and_parameters_key(self):
        t = ('<tool_call>[{"name": "search", "parameters": {"query_list": ["a"]}},'
             '{"name": "search", "arguments": {"query_list": ["b"]}}]</tool_call>')
        calls = A.parse_tool_calls(t)[1]
        self.assertEqual([c["function"]["arguments"]["query_list"] for c in calls], [["a"], ["b"]])

    def test_structure_error(self):
        self.assertEqual(A.parse_tool_calls('<tool_call>["x"]</tool_call>')[2], "structure")


class FormatTest(unittest.TestCase):
    def test_passages_match_upstream_layout(self):
        s = A.format_results([["Paris\nCapital of France.", "Lyon\nCity"], ["Rome\nItaly"]])
        self.assertEqual(json.loads(s)["result"],
                         "Doc 1 (Title: Paris)\nCapital of France.\n\nDoc 2 (Title: Lyon)\nCity"
                         "\n---\nDoc 1 (Title: Rome)\nItaly")
        self.assertEqual(json.loads(A.format_results([]))["result"], "No search results found.")

    def test_permutation_non_identity_and_deterministic(self):
        block = [["a", "b", "c"], ["d", "e", "f"], ["g"]]
        seen = set()
        for s in range(200):
            out, perms = A.permute_results(block, np.random.default_rng(s))
            self.assertNotEqual(perms[0], [0, 1, 2]); self.assertNotEqual(perms[1], [0, 1, 2])
            self.assertEqual(sorted(out[0]), ["a", "b", "c"]); self.assertEqual(out[2], ["g"])
            seen.add(tuple(perms[0]))
        self.assertEqual(len(seen), 5)
        a = A.permute_results(block, np.random.default_rng(7))
        b = A.permute_results(block, np.random.default_rng(7))
        self.assertEqual(a, b)

    def test_em_and_extract(self):
        self.assertEqual(A.extract_pred("x <answer> no </answer> <answer> The Beatles! </answer>"),
                         "The Beatles!")
        self.assertEqual(A.em_check("The Beatles!", ["beatles"]), 1)
        self.assertEqual(A.em_check("no answer here", "beatles"), 0)


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.r = A.Renderer(FakeTokenizer(), [A.TOOL_SCHEMA])

    def test_search_then_answer(self):
        gen, search = scripted([TOOL, "<answer> Paris </answer>"]), fake_search()
        ep = run(A.run_episode(USER, self.r, gen, search))
        self.assertEqual([m["role"] for m in ep.messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(ep.messages[1]["content"], "<think>need</think>")
        self.assertEqual((ep.turns, ep.finish, ep.final_turn_index), (2, "stop", 3))
        self.assertEqual(ep.score(["paris"]), 1)
        self.assertEqual(ep.tool_results[2], [["T1\nA", "T2\nB", "T3\nC"]])
        self.assertEqual(search.calls, [{"query_list": ["q1"]}])
        self.assertTrue(gen.prompts[1][0].endswith("<gen>"))

    def test_max_turns_ends_on_unexecuted_tool_call(self):
        ep = run(A.run_episode(USER, self.r, scripted([TOOL] * 5), fake_search()))
        self.assertEqual((ep.turns, ep.finish), (5, "max_turns"))
        self.assertEqual(ep.messages[-1]["role"], "assistant")
        self.assertEqual(ep.final_turn_index, len(ep.messages) - 1)
        self.assertEqual(ep.score("paris"), 0)

    def test_length_finish(self):
        ep = run(A.run_episode(USER, self.r, scripted([("partial", "length")]), fake_search()))
        self.assertEqual((ep.finish, ep.turns, ep.final_turn_index), ("length", 1, 1))

    def test_max_new_tokens_budget(self):
        gen = scripted(["<answer>x</answer>"])
        run(A.run_episode(USER, self.r, gen, fake_search()))
        prompt, max_new = gen.prompts[0]
        n = len(prompt) // 4 + 1
        self.assertEqual(max_new, min(A.RESPONSE_LEN, A.MAX_MODEL_LEN - n - 1))

    def test_context_full_after_tool_is_unreplayable(self):
        big = [["T\n" + "x" * 20000]]
        ep = run(A.run_episode(USER, self.r, scripted([TOOL]), fake_search(big)))
        self.assertEqual(ep.finish, "stop_after_tool")
        self.assertIsNone(ep.final_turn_index)
        self.assertIsNone(A.replay_prefix(ep, np.random.default_rng(0)))

    def test_bad_tool_call_ends_episode(self):
        ep = run(A.run_episode(USER, self.r, scripted(["<tool_call>{bad}</tool_call>"]), fake_search()))
        self.assertEqual((ep.finish, ep.parse_errors, len(ep.messages)), ("stop", 1, 2))


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.r = A.Renderer(FakeTokenizer(), [A.TOOL_SCHEMA])

    def test_prefix_permutes_passages_and_keeps_budget(self):
        orig = run(A.run_episode(USER, self.r, scripted([TOOL, TOOL, "<answer> Paris </answer>"]),
                                 fake_search()))
        prefix, results, used, stats = A.replay_prefix(orig, np.random.default_rng(3))
        self.assertEqual(len(prefix), orig.final_turn_index)
        self.assertEqual(used, 2)
        self.assertEqual(stats, {"tool_messages": 2, "permuted": 2, "unpermutable": 0})
        for i in (2, 4):
            self.assertNotEqual(prefix[i]["content"], orig.messages[i]["content"])
            self.assertEqual(sorted(results[i][0]), sorted(orig.tool_results[i][0]))
        for i in (0, 1, 3):
            self.assertEqual(prefix[i], orig.messages[i])
        gen = scripted([TOOL, TOOL, TOOL])
        rep = run(A.run_episode(prefix, self.r, gen, fake_search(), turns_used=used, tool_results=results))
        self.assertEqual(rep.turns, 5)          # only 3 turns left
        self.assertEqual(rep.new_tool_calls, 3)

    def test_error_tool_message_is_left_unchanged(self):
        async def failing(args):
            return json.dumps({"result": "Search error: x"}), None
        orig = run(A.run_episode(USER, self.r, scripted([TOOL, "<answer>a</answer>"]), failing))
        prefix, _, _, stats = A.replay_prefix(orig, np.random.default_rng(0))
        self.assertEqual(stats["unpermutable"], 1)
        self.assertEqual(prefix[2], orig.messages[2])

    def test_round_trip_json(self):
        orig = run(A.run_episode(USER, self.r, scripted([TOOL, "<answer>a</answer>"]), fake_search()))
        back = A.episode_from_json(json.loads(json.dumps(orig.to_json())))
        self.assertEqual(back.tool_results, orig.tool_results)
        self.assertEqual(back.final_turn_index, orig.final_turn_index)


@unittest.skipUnless(os.path.isdir(QWEN_TOK), "Qwen2.5 tokenizer not available")
class QwenTemplateTest(unittest.TestCase):
    def test_tool_call_and_response_rendering(self):
        from transformers import AutoTokenizer
        r = A.Renderer(AutoTokenizer.from_pretrained(QWEN_TOK), [A.TOOL_SCHEMA])
        normed, calls, _ = A.parse_tool_calls(TOOL)
        msgs = USER + [A.message("assistant", normed, calls),
                       A.message("tool", A.format_results([["Paris\nCapital"]]))]
        text = r.render(msgs, gen=True)
        self.assertIn('"enum": null', text)
        self.assertIn('"strict": false', text)
        self.assertIn('<think>need</think>\n<tool_call>\n{"name": "search", "arguments": '
                      '{"query_list": ["q1"]}}\n</tool_call><|im_end|>', text)
        self.assertIn("<|im_start|>user\n<tool_response>\n{\"result\": \"Doc 1 (Title: Paris)\\nCapital\"}"
                      "\n</tool_response><|im_end|>", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertGreater(r.ntok(text), 50)


if __name__ == "__main__":
    unittest.main()
