"""Frozen-model paired-world probe for evidence acquisition and use."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

from work3.p1_agent import Renderer, message, parse_tool_calls
from work3.rsi_tasks import Case, SPLITS, make_cases

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Look up an exact inventory record by order ID or SKU.",
        "parameters": {
            "type": "object",
            "properties": {"query_list": {
                "type": "array", "items": {"type": "string"},
                "minItems": 1, "maxItems": 1,
                "description": "One exact query: order ORD-123456 or stock SKU-123456.",
            }},
            "required": ["query_list"],
        },
    },
}]
SYSTEM = """You are an inventory dispatch assistant. Look up one record at a time.
For an order ID, query "order " followed by its ID. For an item, query "stock " followed by its SKU.
Use IDs from the question and returned records. Once the available facts determine the route,
stop querying. End your response with exactly one of the following: a tool call, <answer>A</answer>,
or <answer>B</answer>. A tool call has this form, substituting the current lookup ID:
<tool_call>
{"name":"search","arguments":{"query_list":["stock SKU-000000"]}}
</tool_call>"""
ORACLE_FINISH = ("The lookup phase is complete. Use the records above to decide the route. "
                 "Reply with <answer>A</answer> or <answer>B</answer> and make no further calls.")
ANSWER = re.compile(r"<answer>\s*([AB])\s*</answer>")
QUERY = re.compile(r"(?:order ORD|stock SKU)-[0-9]{6}")
CALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class Completion:
    text: str
    finish: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


Generate = Callable[[list[dict]], Awaitable[Completion]]


@dataclass(frozen=True)
class Decision:
    status: str
    answer: str | None = None
    query: str | None = None
    assistant: dict | None = None
    rejection: str | None = None
    queries: tuple[str, ...] = ()


def decide(completion: Completion, *, max_per_call: int = 1) -> Decision:
    raw = completion.text.strip()
    if completion.finish == "length":
        return Decision("truncated", rejection="completion_limit")
    if "<tool_call" in raw or "</tool_call>" in raw:
        matches = CALL.findall(raw)
        if (len(matches) != 1 or raw.count("<tool_call>") != 1
                or raw.count("</tool_call>") != 1 or not raw.endswith("</tool_call>")
                or "<answer" in raw):
            return Decision("format_error", rejection="malformed_tool_envelope")
        content, calls, error = parse_tool_calls(raw)
        if error or len(calls) != 1:
            return Decision("format_error", rejection="invalid_or_multiple_tool_calls")
        arguments = calls[0]["function"]["arguments"]
        queries = arguments.get("query_list")
        if set(arguments) != {"query_list"} or not isinstance(queries, list):
            return Decision("format_error", rejection="invalid_tool_arguments")
        if len(queries) > max_per_call:
            return Decision("format_error", rejection="multiple_queries")
        if not queries or any(not isinstance(q, str) or QUERY.fullmatch(q) is None for q in queries):
            return Decision("format_error", rejection="invalid_query")
        return Decision("tool", query=queries[0], assistant=message("assistant", content, calls),
                        queries=tuple(queries))
    answers = ANSWER.findall(raw)
    if (len(answers) == 1 and raw.count("<answer>") == 1 and raw.count("</answer>") == 1
            and raw.endswith("</answer>")):
        return Decision("answered", answer=answers[0])
    return Decision("format_error", rejection="missing_or_malformed_answer")


def tool_message(world: dict, query: str) -> dict:
    return message("tool", json.dumps({"record": world.get(query)}, sort_keys=True))


def tool_batch_message(world: dict, queries: tuple[str, ...]) -> dict:
    if len(queries) == 1:
        return tool_message(world, queries[0])
    results = [{"query": query, "record": world.get(query)} for query in queries]
    return message("tool", json.dumps({"results": results}, sort_keys=True))


def oracle_call(query: str) -> dict:
    tool = {"id": "0", "type": "function", "function": {
        "name": "search", "arguments": {"query_list": [query]},
    }}
    return message("assistant", "", [tool])


@dataclass(frozen=True)
class Branch:
    status: str
    answer: str | None
    queries: tuple[str, ...]


def finish(decision: Decision, queries: tuple[str, ...]) -> Branch:
    status = "budget_exhausted" if decision.status == "tool" else decision.status
    return Branch(status, decision.answer, queries)


def initial_history(case: Case) -> list[dict]:
    return [message("system", SYSTEM), message("user", case.prompt)]


def evidence_history(case: Case, world: dict, instruction: str = ORACLE_FINISH) -> list[dict]:
    history = initial_history(case)
    for query in case.oracle:
        history.extend([oracle_call(query), tool_message(world, query)])
    history.append(message("user", instruction))
    return history


async def evaluate(case: Case, generate: Generate, max_queries: int, *,
                   rejections: list[dict] | None = None,
                   oracle_completions: list[Completion] | None = None,
                   allow_batch: bool = False, run_oracle: bool = True,
                   terminals: list[dict] | None = None) -> dict:
    """Evaluate common actions once until the tool result actually differs."""
    initial = initial_history(case)
    costs = {arm: dict(calls=0, prompt_tokens=0, completion_tokens=0, tool_executions=0)
             for arm in ("autonomous", "oracle")}

    async def next_action(history: list[dict], arm: str, worlds: tuple[int, ...]) -> Decision:
        response = await generate(history)
        cost = costs[arm]
        cost["calls"] += 1
        cost["prompt_tokens"] += response.prompt_tokens
        cost["completion_tokens"] += response.completion_tokens
        action = decide(response, max_per_call=max_queries
                        if arm == "autonomous" and allow_batch else 1)
        if arm == "oracle" and oracle_completions is not None:
            oracle_completions.append(response)
        if arm == "autonomous" and rejections is not None and action.rejection:
            rejections.append({
                "worlds": worlds, "reason": action.rejection, "status": action.status,
                "completion": asdict(response), "history": history,
                "executed_queries": sum(len(call["function"]["arguments"]["query_list"])
                                        for msg in history for call in msg["tool_calls"] or []),
            })
        return action

    def lookup(world: dict, queries: tuple[str, ...]) -> dict:
        costs["autonomous"]["tool_executions"] += len(queries)
        return tool_batch_message(world, queries)

    def stopped(action: Decision, queries: tuple[str, ...]) -> bool:
        return action.status != "tool" or len(queries) + len(action.queries) > max_queries

    def end(action: Decision, history: list[dict], queries: tuple[str, ...],
            worlds: tuple[int, ...]) -> Branch:
        branch = finish(action, queries)
        if terminals is not None:
            terminals.append({"worlds": worlds, "history": history, **asdict(branch)})
        return branch

    async def continue_branch(history: list[dict], queries: tuple[str, ...],
                              world_index: int) -> Branch:
        world = case.worlds[world_index]
        while True:
            action = await next_action(history, "autonomous", (world_index,))
            if stopped(action, queries):
                return end(action, history, queries, (world_index,))
            history = history + [action.assistant, lookup(world, action.queries)]
            queries += action.queries

    history = initial
    queries = ()
    fork_at = None
    while True:
        action = await next_action(history, "autonomous", (0, 1))
        if stopped(action, queries):
            branches = (end(action, history, queries, (0, 1)),) * 2
            break
        histories = [history + [action.assistant, lookup(world, action.queries)]
                     for world in case.worlds]
        queries += action.queries
        if histories[0][-1] != histories[1][-1]:
            fork_at = len(queries)
            branches = tuple([await continue_branch(prefix, queries, index)
                              for index, prefix in enumerate(histories)])
            break
        history = histories[0]

    oracle = []
    if not case.control and run_oracle:
        for index, world in enumerate(case.worlds):
            history = evidence_history(case, world)
            costs["oracle"]["tool_executions"] += len(case.oracle)
            action = await next_action(history, "oracle", (index,))
            status = "unexpected_tool" if action.status == "tool" else action.status
            oracle.append(Branch(status, action.answer, case.oracle))

    def correct(marked_branches: tuple[Branch, ...] | list[Branch]) -> list[bool]:
        return [branch.answer == label for branch, label in zip(marked_branches, case.answers)]

    auto_correct = correct(branches)
    oracle_correct = correct(oracle)
    full_evidence = all(set(case.oracle) <= set(branch.queries) for branch in branches)
    sufficient = [case.sufficient(i, branch.queries) for i, branch in enumerate(branches)]
    all_errors_had_evidence = all(had_evidence or is_correct
                                  for had_evidence, is_correct in zip(sufficient, auto_correct))
    any_error_had_evidence = any(had_evidence and branch.status == "answered" and not is_correct
                                 for had_evidence, is_correct, branch in
                                 zip(sufficient, auto_correct, branches))
    if case.control:
        diagnosis = "control"
    elif all(auto_correct):
        diagnosis = "autonomous_solved"
    elif any(branch.status in {"format_error", "truncated"} for branch in branches):
        diagnosis = "autonomous_format_or_truncation"
    elif all(branch.status == "answered" for branch in branches) and all_errors_had_evidence:
        diagnosis = "post_observation_error"
    elif any_error_had_evidence:
        diagnosis = "mixed_or_unresolved_after_observation"
    elif any(branch.status == "budget_exhausted" and had_evidence
             for branch, had_evidence in zip(branches, sufficient)):
        diagnosis = "budget_exhausted_after_observation"
    elif not run_oracle:
        diagnosis = "evidence_intervention_pending"
    elif any(branch.status != "answered" for branch in oracle):
        diagnosis = "oracle_format_or_truncation"
    elif not all(oracle_correct):
        diagnosis = "oracle_incorrect"
    elif fork_at is None:
        diagnosis = "decisive_evidence_not_acquired"
    else:
        diagnosis = "other_required_evidence_not_acquired"

    return {
        "case_id": case.id, "family": case.family, "control": case.control,
        "labels": case.answers, "autonomous": [asdict(branch) for branch in branches],
        "oracle": [asdict(branch) for branch in oracle],
        "autonomous_correct": auto_correct, "oracle_correct": oracle_correct,
        "fork_at_query": fork_at, "full_evidence_both": full_evidence and not case.control,
        "autonomous_sufficient_evidence": sufficient,
        "diagnosis": diagnosis, "costs": costs,
    }


def metrics(rows: list[dict]) -> dict:
    targets = [row for row in rows if not row["control"]]
    controls = [row for row in rows if row["control"]]

    def tally(arm: str, pool: list[dict]) -> dict:
        return {
            "pairs": len(pool),
            "pairs_correct": sum(all(row[f"{arm}_correct"]) for row in pool),
            "worlds_correct": sum(sum(row[f"{arm}_correct"]) for row in pool),
            "worlds_with_answer": sum(branch["status"] == "answered"
                                      for row in pool for branch in row[arm]),
            "statuses": counts(branch["status"] for row in pool for branch in row[arm]),
        }

    return {
        "autonomous": tally("autonomous", targets),
        "oracle": tally("oracle", targets),
        "diagnoses": counts(row["diagnosis"] for row in targets),
        "decisive_evidence_pairs": sum(row["fork_at_query"] is not None for row in targets),
        "full_evidence_pairs": sum(row["full_evidence_both"] for row in targets),
        "controls": {**tally("autonomous", controls),
                     "pairs_with_tool": sum(any(branch["queries"] for branch in row["autonomous"])
                                            for row in controls)},
        "costs": {arm: {key: sum(row["costs"][arm][key] for row in rows)
                        for key in ("calls", "prompt_tokens", "completion_tokens", "tool_executions")}
                  for arm in ("autonomous", "oracle")},
    }


def counts(values) -> dict:
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


async def request_completion(client, renderer: Renderer, endpoint: str, model: str,
                             max_tokens: int, history: list[dict]) -> Completion:
    response = await client.post(endpoint.rstrip("/") + "/v1/completions", json={
        "model": model, "prompt": renderer.render(history, gen=True),
        "temperature": 0, "max_tokens": max_tokens,
    })
    if response.is_error:
        raise RuntimeError(f"Completion endpoint returned {response.status_code}: "
                           f"{response.text[:500]}")
    body = response.json()
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return Completion(choice["text"], choice.get("finish_reason") or "unknown",
                      usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


async def run(args: argparse.Namespace) -> dict:
    import httpx
    from transformers import AutoTokenizer

    cases = make_cases(args.split, args.cases_per_family, args.controls_per_family, args.seed)
    root = Path(__file__).resolve().parent.parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                         capture_output=True, check=False)
    provenance = {"revision": git.stdout.strip() or None, "protocol": 2, **vars(args)}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = Renderer(tokenizer, TOOLS)
    limiter = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        async def generate(history: list[dict]) -> Completion:
            async with limiter:
                return await request_completion(client, renderer, args.endpoint, args.model,
                                                args.max_tokens, history)

        rows = []
        for start in range(0, len(cases), args.concurrency):
            batch = cases[start:start + args.concurrency]
            rows.extend(await asyncio.gather(
                *(evaluate(case, generate, args.max_queries) for case in batch)))
            print(f"Completed {len(rows)}/{len(cases)} pairs", flush=True)
    result = {
        "config": {**provenance, "temperature": 0},
        "overall": metrics(rows),
        "by_family": {family: metrics([row for row in rows if row["family"] == family])
                      for family in SPLITS[args.split]},
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "paired_rsi_cases.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = output / "paired_rsi_summary.json"
    temporary = summary.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8187")
    parser.add_argument("--model", required=True, help="Model name served by the completion endpoint")
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer path")
    parser.add_argument("--split", choices=tuple(SPLITS), default="heldout")
    parser.add_argument("--out-dir", required=True, help="New directory for this run")
    parser.add_argument("--cases-per-family", type=int, default=60)
    parser.add_argument("--controls-per-family", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--max-queries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if (args.cases_per_family < 1 or args.controls_per_family < 0 or args.max_queries < 3
            or min(args.max_tokens, args.concurrency) < 1):
        parser.error("Use positive case/token/concurrency counts, >=0 controls and >=3 queries.")
    if Path(args.out_dir).exists():
        parser.error("The output directory must not exist.")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
