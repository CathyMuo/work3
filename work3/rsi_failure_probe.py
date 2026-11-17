"""Frozen paired-world batch-interface and same-history evidence diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
from pathlib import Path

from work3.rsi_calibrate import COMPARE, MAX_QUERIES, MAX_TOKENS, diagnostic_cases, score_output
from work3.rsi_probe import TOOLS, Generate, Renderer, counts, evaluate, message, request_completion
from work3.rsi_tasks import Case, make_case

SEED = 20261119
FRESH_CASES = 30
FRESH_CONTROLS = 8
FAMILIES = ("amended", "pool")


def linked_case(family: str, index: int, *, placebo: bool = False) -> Case:
    """Create a new synthetic case or independent, format-matched placebo records."""
    rng = random.Random(f"{SEED}:{family}:{index}:{placebo}")
    ids = rng.sample(range(500000 if placebo else 100000,
                           900000 if placebo else 500000), 5)
    root, active, old = (f"ORD-{serial}" for serial in ids[:3])
    first, second = (f"SKU-{serial}" for serial in ids[3:])
    order, current, archived = f"order {root}", f"order {active}", f"order {old}"
    stock1, stock2 = f"stock {first}", f"stock {second}"
    quantity = rng.randint(12, 19)
    if family == "amended":
        prompt = (f"Ticket {root} points to the current and superseded versions of an order. "
                  "Use the current version only. Route to A if its SKU's on-hand stock covers "
                  "its required quantity, otherwise route to B.")
        records = {order: {"current_order": active, "superseded_order": old},
                   current: {"sku": first, "quantity": quantity},
                   archived: {"sku": second, "quantity": quantity - 1},
                   stock2: {"on_hand": quantity + 2, "reserved": 0}}
        plan, changed = (order, current, stock1), stock1
        base = quantity
    else:
        prompt = (f"Order {root} names two approved, interchangeable lots of an item. "
                  "Both lots may be combined. Route to A if their combined on-hand stock covers "
                  "the order's required quantity, otherwise route to B.")
        on_hand = rng.randint(4, 8)
        records = {order: {"lot_a": first, "lot_b": second, "quantity": quantity},
                   stock1: {"on_hand": on_hand, "reserved": 0}}
        plan, changed = (order, stock1, stock2), stock2
        base = quantity - on_hand
    low, high = base - rng.randint(1, 2), base + rng.randint(0, 2)
    levels = (high, low) if rng.randrange(2) else (low, high)
    worlds = tuple({**records, changed: {"on_hand": n, "reserved": 0}} for n in levels)
    answers = tuple("A" if world[changed]["on_hand"] >= base else "B" for world in worlds)
    return Case(f"{family}-{index}", family, False, prompt, worlds, plan, changed, answers)


async def run_batch(case: Case, cohort: str, generate: Generate) -> dict:
    arms = {}
    completions = {}

    async def shared_generate(history: list[dict]):
        key = json.dumps(history, sort_keys=True)
        if key not in completions:
            completions[key] = await generate(history)
        return completions[key]

    for name, allow in (("single", False), ("batch", True)):
        rejections: list[dict] = []
        row = await evaluate(case, shared_generate, MAX_QUERIES, rejections=rejections,
                             allow_batch=allow, run_oracle=False)
        arms[name] = {
            "branches": row["autonomous"], "correct": row["autonomous_correct"],
            "fork_at_query": row["fork_at_query"], "full_evidence_both": row["full_evidence_both"],
            "sufficient_evidence": row["autonomous_sufficient_evidence"],
            "cost": row["costs"]["autonomous"], "rejections": rejections,
        }
    return {"case_id": case.id, "cohort": cohort, "control": case.control,
            "labels": case.answers, "arms": arms, "unique_generation_calls": len(completions)}


def supplement(world: dict, queries: list[str]) -> dict:
    records = [{"query": query, "record": world[query]} for query in queries]
    text = "Supplemental lookup records:\n" + json.dumps(records, sort_keys=True)
    return message("user", text + "\n\n" + COMPARE)


async def run_evidence(case: Case, placebo: Case, generate: Generate) -> dict:
    terminals: list[dict] = []
    row = await evaluate(case, generate, MAX_QUERIES, allow_batch=True, run_oracle=False,
                         terminals=terminals)
    worlds = []
    for terminal in terminals:
        for i in terminal["worlds"]:
            status = terminal["status"]
            result = {"world": i, "label": case.answers[i], "answer": terminal["answer"],
                      "status": status, "queries": terminal["queries"]}
            missing = [j for j, key in enumerate(case.oracle) if key not in terminal["queries"]]
            result["missing"] = [case.oracle[j] for j in missing]
            if status != "answered":
                category = status
            elif terminal["answer"] == case.answers[i]:
                category = "autonomous_correct"
            elif not missing:
                category = "wrong_with_sufficient_evidence"
            else:
                result["history"] = terminal["history"]
                outputs = {}
                arms = [("relevant", case), ("placebo", placebo)]
                if (int(case.id.rsplit("-", 1)[1]) + i) % 2:
                    arms.reverse()
                for arm, source in arms:
                    addon = supplement(source.worlds[i], [source.oracle[j] for j in missing])
                    outputs[arm] = {
                        "message": addon,
                        **score_output(await generate(terminal["history"] + [addon]), case.answers[i]),
                    }
                result["intervention"] = outputs
                if any(output["status"] != "answered" for output in outputs.values()):
                    category = "invalid_intervention_output"
                elif outputs["relevant"]["correct"] and not outputs["placebo"]["correct"]:
                    category = "rescued_only_by_relevant_evidence"
                elif outputs["relevant"]["correct"] or outputs["placebo"]["correct"]:
                    category = "rerun_correct_without_specific_rescue"
                else:
                    category = "neither_rerun_correct"
            result["category"] = category
            worlds.append(result)
    return {"case_id": case.id, "family": case.family, "fork_at_query": row["fork_at_query"],
            "worlds": sorted(worlds, key=lambda w: w["world"])}


def batch_metrics(rows: list[dict], arm: str) -> dict:
    states = [row["arms"][arm] for row in rows]
    return {
        "pairs": len(states),
        "pairs_correct": sum(all(state["correct"]) for state in states),
        "pairs_with_two_answers": sum(all(b["status"] == "answered" for b in state["branches"])
                                      for state in states),
        "pairs_with_differing_tool_return": sum(state["fork_at_query"] is not None for state in states),
        "pairs_with_full_evidence": sum(state["full_evidence_both"] for state in states),
        "pairs_with_tool": sum(any(b["queries"] for b in state["branches"]) for state in states),
        "rejection_reasons": counts(r["reason"] for state in states for r in state["rejections"]),
        "queries": sum(state["cost"]["tool_executions"] for state in states),
        "generation_calls": sum(state["cost"]["calls"] for state in states),
    }


def batch_summary(rows: list[dict]) -> dict:
    report = {}
    for cohort in ("replay", "fresh"):
        report[cohort] = {}
        for control, name in ((False, "targets"), (True, "controls")):
            group = [r for r in rows if r["cohort"] == cohort and r["control"] == control]
            strict = batch_metrics(group, "single")
            batch = batch_metrics(group, "batch")
            recovered = [row for row in group if any(
                event["reason"] == "multiple_queries" for event in row["arms"]["single"]["rejections"])]
            report[cohort][name] = {
                "single": strict, "batch": batch, "strict_multi_query_rejections": len(recovered),
                "unique_generation_calls": sum(row["unique_generation_calls"] for row in group),
                "rejected_pairs_reaching_evidence_in_batch": sum(
                    row["arms"]["batch"]["fork_at_query"] is not None for row in recovered),
                "rejected_pairs_correct_in_batch": sum(
                    all(row["arms"]["batch"]["correct"]) for row in recovered),
                "batch_pair_wins": sum(all(r["arms"]["batch"]["correct"]) and
                                       not all(r["arms"]["single"]["correct"]) for r in group),
                "batch_pair_losses": sum(all(r["arms"]["single"]["correct"]) and
                                         not all(r["arms"]["batch"]["correct"]) for r in group),
            }
    return report


def evidence_summary(rows: list[dict]) -> dict:
    result = {}
    for family in FAMILIES:
        pool = [row for row in rows if row["family"] == family]
        result[family] = {
            "pairs": len(pool),
            "world_categories": counts(w["category"] for row in pool for w in row["worlds"]),
            "pairs_with_intervention": sum(any("intervention" in w for w in row["worlds"])
                                          for row in pool),
            "pairs_with_relevant_only_rescue": sum(any(
                w["category"] == "rescued_only_by_relevant_evidence" for w in row["worlds"])
                for row in pool),
            "pairs_with_sufficient_evidence_error": sum(any(
                w["category"] == "wrong_with_sufficient_evidence" for w in row["worlds"])
                for row in pool),
        }
    return result


async def run(args: argparse.Namespace) -> dict:
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = Renderer(tokenizer, TOOLS)
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if args.phase == "batch":
        cases = [(case, "replay") for case, _ in diagnostic_cases() if case.family == "bundle"]
        cases += [(make_case("bundle", SEED, i, control), "fresh")
                  for control, n in ((False, FRESH_CASES), (True, FRESH_CONTROLS)) for i in range(n)]
    else:
        cases = [(linked_case(family, i), linked_case(family, i, placebo=True))
                 for family in FAMILIES for i in range(FRESH_CASES)]

    rows = []
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        async def generate(history: list[dict]):
            return await request_completion(client, renderer, args.endpoint, args.model,
                                            MAX_TOKENS, history)

        for start in range(0, len(cases), args.concurrency):
            chunk = cases[start:start + args.concurrency]
            if args.phase == "batch":
                tasks = (run_batch(case, cohort, generate) for case, cohort in chunk)
            else:
                tasks = (run_evidence(case, placebo, generate) for case, placebo in chunk)
            rows.extend(await asyncio.gather(*tasks))
            print(f"Completed {len(rows)}/{len(cases)} pairs", flush=True)
    result = {
        "config": {"protocol": "rsi_failure_probe_v1", "revision": revision, "seed": SEED,
                   "temperature": 0, "max_queries": MAX_QUERIES, "max_tokens": MAX_TOKENS,
                   **vars(args)},
        "total_pairs": len(rows),
        "metrics": batch_summary(rows) if args.phase == "batch" else evidence_summary(rows),
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rsi_failure_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary = output / "rsi_failure_summary.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "rsi_failure_summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("batch", "evidence"), required=True)
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
