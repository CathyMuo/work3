"""Test whether repairing one evidence lookup transfers to new order-link structures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
from dataclasses import replace
from pathlib import Path

from work3.rsi_calibrate import MAX_QUERIES, MAX_TOKENS
from work3.rsi_failure_probe import linked_case
from work3.rsi_probe import TOOLS, Completion, Generate, Renderer, counts, decide, evaluate, request_completion
from work3.rsi_tasks import Case

SEED = 20261121
FAMILY_COUNTS = {"amended": 30, "status_list": 60, "revision_list": 60}
ARMS = ("autonomous", "grounded", "force_current", "force_archived")
GROUNDING = ("Use only exact IDs present in the question or tool results. "
             "When a record points to another relevant record, fetch that record before "
             "looking up fields it may contain. Never invent an ID.")


def pointer_case(family: str, index: int) -> tuple[Case, str]:
    case = linked_case("amended", index, seed=SEED * 3 + tuple(FAMILY_COUNTS).index(family))
    root_query, current_query, _ = case.oracle
    root_id = root_query.split()[1]
    root = case.worlds[0][root_query]
    archived_id = root["superseded_order"]
    archived_query = "order " + archived_id
    if family == "amended":
        return case, archived_query
    current_id = current_query.split()[1]
    rng = random.Random(f"pointer:{SEED}:{family}:{index}")
    if family == "status_list":
        versions = [{"order_id": current_id, "status": "approved"},
                    {"order_id": archived_id, "status": "withdrawn"}]
        rule = "Use the only version marked approved."
    elif family == "revision_list":
        low = rng.randint(1, 8)
        versions = [{"order_id": current_id, "revision": low + 1},
                    {"order_id": archived_id, "revision": low}]
        rule = "Use the version with the highest revision number."
    else:
        raise ValueError(f"Unknown family: {family}")
    rng.shuffle(versions)
    prompt = (f"Order ticket {root_id} lists two versions of the order. {rule} "
              "Route to A if the on-hand stock covers that version's required quantity, "
              "otherwise route to B.")
    worlds = tuple({**world, root_query: {"versions": versions}} for world in case.worlds)
    return replace(case, id=f"{family}-{index}", family=family, prompt=prompt, worlds=worlds), archived_query


async def probe_case(case: Case, archived: str, generate: Generate) -> dict:
    results = {}
    index = int(case.id.rsplit("-", 1)[1])
    arms = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
    for arm in arms:
        root, current, stock = case.oracle
        extra = {"force_current": (current,), "force_archived": (archived,)}.get(arm, ())
        prefix = (root,) + extra
        rejects: list[dict] = []
        first: Completion | None = None

        async def capture(history):
            nonlocal first
            completion = await generate(history)
            if first is None:
                first = completion
            return completion

        out = await evaluate(
            case, capture, MAX_QUERIES, allow_batch=True, run_oracle=False, rejections=rejects,
            allow_answer_period=True, allow_bare_answer=True,
            seed_queries=prefix, extra_system_instruction=GROUNDING if arm == "grounded" else "",
        )
        assert first is not None
        action = decide(first, max_per_call=MAX_QUERIES, allow_answer_period=True,
                        allow_bare_answer=True)
        branches = out["autonomous"]
        results[arm] = {
            "prefix_queries": prefix,
            "first_status": action.status,
            "first_completion": first.text,
            "first_queries": action.queries,
            "branches": branches,
            "pair_correct": all(out["autonomous_correct"]),
            "both_valid": all(branch["status"] == "answered" for branch in branches),
            "current_in_both_worlds": all(current in branch["queries"] for branch in branches),
            "decisive_stock_in_both_worlds": all(stock in branch["queries"] for branch in branches),
            "null_lookup_in_either_world": any(q not in world for world, branch in
                                               zip(case.worlds, branches) for q in branch["queries"]),
            "fork_at_query": out["fork_at_query"],
            "cost": out["costs"]["autonomous"],
            "rejections": rejects,
        }
    return {"case_id": case.id, "family": case.family, "labels": case.answers,
            "required_queries": case.oracle, "archived_query": archived, "arms": results}


def summarize(rows: list[dict]) -> dict:
    metrics = {}
    for family in FAMILY_COUNTS:
        cases = [row for row in rows if row["family"] == family]
        per_arm = {}
        for arm in ARMS:
            items = [r["arms"][arm] for r in cases]
            per_arm[arm] = {
                "pairs": len(cases),
                **{key: sum(item[key] for item in items) for key in (
                    "pair_correct", "both_valid", "current_in_both_worlds",
                    "decisive_stock_in_both_worlds", "null_lookup_in_either_world",
                )},
                "first_action_contains_current": sum(
                    row["required_queries"][1] in row["arms"][arm]["first_queries"] for row in cases),
                "first_action_contains_archived": sum(
                    row["archived_query"] in row["arms"][arm]["first_queries"] for row in cases),
                "generation_calls": sum(item["cost"]["calls"] for item in items),
                "prompt_tokens": sum(item["cost"]["prompt_tokens"] for item in items),
                "completion_tokens": sum(item["cost"]["completion_tokens"] for item in items),
                "queries": sum(item["cost"]["tool_executions"] for item in items),
                "rejections": counts(e["reason"] for item in items for e in item["rejections"]),
            }

        def matched(chosen: str, reference: str) -> dict:
            pairs = [(r["arms"][chosen]["pair_correct"], r["arms"][reference]["pair_correct"])
                     for r in cases]
            return {"wins": sum(a and not b for a, b in pairs),
                    "losses": sum(b and not a for a, b in pairs)}

        control = matched("force_current", "force_archived")
        headroom = sum(
            r["arms"]["force_current"]["pair_correct"]
            and r["arms"]["force_archived"]["both_valid"]
            and not r["arms"]["force_archived"]["pair_correct"]
            and r["arms"]["grounded"]["both_valid"]
            and not r["arms"]["grounded"]["pair_correct"]
            and not r["arms"]["grounded"]["decisive_stock_in_both_worlds"] for r in cases)
        metrics[family] = {
            "arms": per_arm,
            "grounded_vs_autonomous": matched("grounded", "autonomous"),
            "current_vs_archived": control,
            "current_vs_grounded": matched("force_current", "grounded"),
            "selective_retrieval_headroom_pairs": headroom,
            "screen": control["wins"] - control["losses"] >= 10 and headroom >= 6
            if family != "amended" else None,
        }
    return {
        "by_family": metrics,
        "acquisition_learning_screen_pass": all(metrics[family]["screen"]
                                                 for family in FAMILY_COUNTS if family != "amended"),
    }


async def run(args: argparse.Namespace) -> dict:
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = Renderer(tokenizer, TOOLS)
    folder = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=folder, text=True).strip()
    cases = [pointer_case(family, i) for family, n in FAMILY_COUNTS.items() for i in range(n)]
    rows = []
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        async def generate(history: list[dict]) -> Completion:
            return await request_completion(client, renderer, args.endpoint, args.model,
                                            MAX_TOKENS, history)

        for start in range(0, len(cases), args.concurrency):
            batch = cases[start:start + args.concurrency]
            rows.extend(await asyncio.gather(*(probe_case(case, archived, generate)
                                               for case, archived in batch)))
            print(f"Completed {len(rows)}/{len(cases)} pairs", flush=True)
    report = {
        "config": {**vars(args), "revision": revision, "protocol": "rsi_pointer_probe_v1",
                   "temperature": 0, "seed": SEED, "max_queries": MAX_QUERIES,
                   "max_tokens": MAX_TOKENS},
        "total_pairs": len(rows), **summarize(rows),
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rsi_pointer_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary = output / "rsi_pointer_summary.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "rsi_pointer_summary.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8187")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or Path(args.out_dir).exists():
        parser.error("Use positive concurrency and a new output directory.")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
