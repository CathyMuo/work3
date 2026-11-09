"""Solver episodes and same-prefix replays for the P1 diagnosis.

Mirrors Dr. Zero's proposer-reward rollout at the pinned commit
(`verl/custom_reward/reward_rollout.py` + `reward_function.py`):

- prompt = tokenizer chat template over the message list with the search tool
  schema, generation prompt appended, sent as text to SGLang /v1/completions;
- max 5 assistant turns, max_model_len 3072 (prompt 512 + response 2560),
  per-turn max_new_tokens = min(2560, 3072 - len(prompt) - 1), temperature 1.0,
  top_p 1.0, presence / frequency penalty 0;
- tool calls parsed like SGLang 0.4.6.post5 Qwen25Detector + verl's
  `from_openai_function_parsed_schema`; the assistant message keeps the
  normalised text and structured tool calls (re-rendered by the chat template);
- tool responses formatted like verl `search_r1_like_utils` (top-3 passages,
  "Doc i (Title: ...)", queries joined by "\\n---\\n", JSON {"result": ...});
- score = EM of the last <answer> of the last message against the proposer answer.

Upstream builds prompts token-incrementally; this module re-renders the whole
conversation each turn. `p1_parity` checks the two agree on the factory node.

Replays keep every message before the final assistant turn, apply an
independent non-identity permutation to the passages of each query inside every
tool response (passage text unchanged, "Doc i" labels follow the new order), and
re-generate from the start of the final assistant turn with the remaining turn
budget (P1 protocol, Rollouts step 3).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import string
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Tuple

MAX_TURNS = 5
PROMPT_LEN = 512
RESPONSE_LEN = 2560
MAX_MODEL_LEN = PROMPT_LEN + RESPONSE_LEN
TEMPERATURE = 1.0
TOP_P = 1.0
TOPK = 3

ANSWER_PATTERN = r"<answer>(.*?)</answer>"
BOT, EOT = "<tool_call>", "</tool_call>"
TOOL_NAME = "search"

# verl OpenAIFunctionToolSchema(...).model_dump() of Dr. Zero's
# config/search_tool_config.yaml (the YAML's misspelt "item" key is dropped by
# pydantic; "enum": None and "strict": False are added). Field order matters
# because the chat template serialises it.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "description": ("A list of fully-formed semantic queries. The tool will "
                                    "return search results for each query."),
                    "enum": None,
                }
            },
            "required": ["query_list"],
        },
        "strict": False,
    },
}


def upstream_tool_schemas(tool_config_path: str | None = None) -> tuple[list[dict], str]:
    """Return (tools, source). Uses verl's pydantic dump when importable."""
    if tool_config_path:
        try:
            import yaml
            from verl.tools.schemas import OpenAIFunctionToolSchema

            with open(tool_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            tools = [OpenAIFunctionToolSchema.model_validate(t["tool_schema"]).model_dump()
                     for t in cfg["tools"]]
            return tools, "verl"
        except Exception:  # noqa: BLE001 - fall back to the mirrored constant
            pass
    return [json.loads(json.dumps(TOOL_SCHEMA))], "mirror"


# ---------------------------------------------------------------- scoring
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred = normalize_answer(prediction)
    return int(any(normalize_answer(g) == pred for g in golden_answers))


def extract_pred(last_content: Any) -> str:
    text = last_content if isinstance(last_content, str) else json.dumps(last_content)
    found = re.findall(ANSWER_PATTERN, text, re.DOTALL)
    return found[-1].strip() if found else text.strip()


# ---------------------------------------------------------------- tool calls
def parse_tool_calls(content: str) -> tuple[str, list[dict], str | None]:
    """(normed_content, tool_calls, error). Only called when BOT in content."""
    error = None
    items: list[tuple[str, str]] = []
    normal = content[: content.find(BOT)].strip()
    try:
        for match in re.findall(rf"{BOT}(.*?){EOT}", content, re.DOTALL):
            obj = json.loads(match)
            acts = obj if isinstance(obj, list) else [obj]
            for act in acts:
                name = act.get("name")
                if name and name == TOOL_NAME:
                    items.append((name, json.dumps(act.get("parameters") or act.get("arguments", {}),
                                                   ensure_ascii=False)))
    except json.JSONDecodeError:
        items, error = [], "json"
    except (AttributeError, TypeError):
        items, error = [], "structure"
    normed = normal if items else content
    calls = []
    for name, params in items:
        try:
            args = json.loads(params)
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        calls.append({"id": "0", "type": "function", "function": {"name": name, "arguments": args}})
    return normed, calls, error


# ---------------------------------------------------------------- search results
def passages_to_string(passages: list[str]) -> str:
    out = ""
    for idx, content in enumerate(passages):
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return out.strip()


def format_results(results: list[list[str]] | None, message: str | None = None) -> str:
    if results is None:
        return json.dumps({"result": message or "Search request failed or timed out after retries."})
    if not results:
        return json.dumps({"result": "No search results found."})
    return json.dumps({"result": "\n---\n".join(passages_to_string(r) for r in results)})


NON_IDENTITY_3 = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]


def permute_results(results: list[list[str]], rng) -> tuple[list[list[str]], list[list[int]]]:
    """Independent uniformly random non-identity permutation per query block."""
    out, perms = [], []
    for block in results:
        n = len(block)
        if n < 2:
            out.append(list(block)); perms.append(list(range(n)))
            continue
        cands = NON_IDENTITY_3 if n == 3 else [p for p in itertools.permutations(range(n))
                                                if p != tuple(range(n))]
        perm = list(cands[int(rng.integers(len(cands)))])
        out.append([block[i] for i in perm]); perms.append(perm)
    return out, perms


# ---------------------------------------------------------------- episode
# typing aliases evaluated at import time: keep them valid on Python 3.9 (factory feedback #2)
Generate = Callable[[str, int], Awaitable[Tuple[str, str]]]          # (prompt, max_new) -> (text, finish)
Search = Callable[[Any], Awaitable[Tuple[str, Optional[List[List[str]]]]]]  # (arguments) -> (content, passages)


@dataclass
class Episode:
    messages: list[dict]
    tool_results: dict[int, list[list[str]] | None] = field(default_factory=dict)  # msg index -> passages
    finish: str = ""
    turns: int = 0
    final_prompt: str | None = None
    final_turn_index: int | None = None   # index of the last assistant message, if generated last
    parse_errors: int = 0
    new_tool_calls: int = 0                # tool calls issued during this run (replays: new searches)

    def score(self, answer) -> int:
        return em_check(extract_pred(self.messages[-1]["content"]), answer)

    def to_json(self) -> dict:
        return {"messages": self.messages,
                "tool_results": {str(k): v for k, v in self.tool_results.items()},
                "finish": self.finish, "turns": self.turns, "final_prompt": self.final_prompt,
                "final_turn_index": self.final_turn_index, "parse_errors": self.parse_errors,
                "new_tool_calls": self.new_tool_calls}


def message(role: str, content: Any, tool_calls: list | None = None) -> dict:
    return {"role": role, "content": content, "tool_calls": tool_calls}


class Renderer:
    def __init__(self, tokenizer, tools: list[dict]):
        self.tok, self.tools = tokenizer, tools

    def render(self, messages: list[dict], gen: bool) -> str:
        return self.tok.apply_chat_template(messages, tools=self.tools,
                                            add_generation_prompt=gen, tokenize=False)

    def ntok(self, text: str) -> int:
        return len(self.tok(text=[text])["input_ids"][0])


async def run_episode(messages: list[dict], renderer: Renderer, generate: Generate, search: Search,
                      turns_used: int = 0, tool_results: dict | None = None) -> Episode:
    """Continue a conversation like upstream `_async_rollout_a_request`."""
    ep = Episode(messages=[dict(m) for m in messages], tool_results=dict(tool_results or {}),
                 turns=turns_used)
    state = "RUNNING"
    finish = None
    while ep.turns < MAX_TURNS:
        if state == "TOOL_CALLING":
            calls = ep.messages[-1]["tool_calls"]
            outs = await asyncio.gather(*[search(c["function"]["arguments"]) for c in calls])
            for content, passages in outs:
                ep.messages.append(message("tool", content))
                ep.tool_results[len(ep.messages) - 1] = passages
            if renderer.ntok(renderer.render(ep.messages, gen=False)) >= MAX_MODEL_LEN:
                finish = "stop_after_tool"
                break
            state = "RUNNING"
            continue
        prompt = renderer.render(ep.messages, gen=True)
        n = renderer.ntok(prompt)
        if n + 1 >= MAX_MODEL_LEN:
            finish = "length_before_generation"
            break
        text, reason = await generate(prompt, min(RESPONSE_LEN, MAX_MODEL_LEN - n - 1))
        ep.turns += 1
        ep.final_prompt = prompt
        if reason == "length":
            ep.messages.append(message("assistant", text))
            finish = "length"
            break
        if BOT in text:
            normed, calls, err = parse_tool_calls(text)
            ep.parse_errors += err is not None
            if calls:
                ep.messages.append(message("assistant", normed, calls))
                ep.new_tool_calls += len(calls)
                state = "TOOL_CALLING"
                finish = "tool_call"
                continue
            ep.messages.append(message("assistant", text))
            finish = "stop"
            break
        ep.messages.append(message("assistant", text))
        finish = "stop"
        break
    if ep.turns >= MAX_TURNS and finish in (None, "tool_call"):
        finish = "max_turns"
    ep.finish = finish or "stop"
    last = len(ep.messages) - 1
    ep.final_turn_index = last if ep.messages[last]["role"] == "assistant" and ep.final_prompt else None
    return ep


def replay_prefix(original: Episode, rng) -> tuple[list[dict], dict, int, dict] | None:
    """Prefix before the final assistant turn with permuted passages, or None.

    Returns (messages, tool_results, turns_used, stats).
    """
    f = original.final_turn_index
    if f is None:
        return None
    prefix = [dict(m) for m in original.messages[:f]]
    results = {}
    stats = {"tool_messages": 0, "permuted": 0, "unpermutable": 0}
    for i, m in enumerate(prefix):
        if m["role"] != "tool":
            continue
        stats["tool_messages"] += 1
        passages = original.tool_results.get(i)
        if not passages:
            stats["unpermutable"] += 1
            results[i] = passages
            continue
        new, _ = permute_results(passages, rng)
        prefix[i] = message("tool", format_results(new))
        results[i] = new
        stats["permuted"] += 1
    turns_used = sum(1 for m in prefix if m["role"] == "assistant")
    return prefix, results, turns_used, stats


def episode_from_json(d: dict) -> Episode:
    return Episode(messages=d["messages"],
                   tool_results={int(k): v for k, v in d.get("tool_results", {}).items()},
                   finish=d.get("finish", ""), turns=d.get("turns", 0),
                   final_prompt=d.get("final_prompt"), final_turn_index=d.get("final_turn_index"),
                   parse_errors=d.get("parse_errors", 0), new_tool_calls=d.get("new_tool_calls", 0))
