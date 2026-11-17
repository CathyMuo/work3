"""Audit access and evidence use with matched finalization on fresh paired tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

from work3.rsi_calibrate import COMPARE, MAX_QUERIES, MAX_TOKENS, score_output
from work3.rsi_failure_probe import FAMILIES, linked_case
from work3.rsi_probe import TOOLS, Generate, Renderer, counts, evaluate, message, request_completion
from work3.rsi_tasks import Case

SEED = 20261120
PAIRS_PER_FAMILY = 60
ACQUISITION_ARMS = ("relevant", "placebo", "empty")
POOL_ARMS = ("unassisted", "verified_subtotal")
QUANTITIES = re.compile(r"<quantities>(.*?)</quantities>", re.DOTALL)
POOL_INSTRUCTION = (
    "Use the retrieved order and stock records. Report the required quantity, on-hand quantities "
    "for the two lots in order-record order, and their total in one "
    '<quantities>{"required": 0, "lot_a": 0, "lot_b": 0, "total": 0}</quantities> block. '
    "Those zeroes are placeholders. Apply the routing rule, ending with "
    "<answer>A</answer> or <answer>B</answer>. Do not call tools."
)


def arm_order(arms: tuple[str, ...], case: Case, world: int) -> tuple[str, ...]:
    rotation = (int(case.id.rsplit("-", 1)[1]) + world) % len(arms)
    return arms[rotation:] + arms[:rotation]


def supplement(case: Case, world: int, missing: list[int]) -> list[dict]:
    return [{"query": case.oracle[j], "record": case.worlds[world][case.oracle[j]]}
            for j in missing]


async def acquisition_replay(case: Case, placebo: Case, world: int, history: list[dict],
                             missing: list[int], generate: Generate) -> dict:
    records = {"relevant": supplement(case, world, missing),
               "placebo": supplement(placebo, world, missing), "empty": []}
    outputs = {}
    for arm in arm_order(ACQUISITION_ARMS, case, world):
        addon = message("user", "Supplemental lookup records:\n"
                        + json.dumps(records[arm], sort_keys=True) + "\n\n" + COMPARE)
        response = await generate(history + [addon])
        outputs[arm] = {"message": addon, **score_output(
            response, case.answers[world], allow_answer_period=True)}
    both_controls_fail = all(not outputs[arm]["correct"] for arm in ("placebo", "empty"))
    return {
        "arms": outputs,
        "all_valid": all(output["status"] == "answered" for output in outputs.values()),
        "relevant_only": outputs["relevant"]["correct"] and both_controls_fail
        and all(output["status"] == "answered" for output in outputs.values()),
    }


def pool_numbers(case: Case, world: int) -> dict:
    records = case.worlds[world]
    order = records[case.oracle[0]]
    lot_a, lot_b = (records["stock " + order[lot]]["on_hand"] for lot in ("lot_a", "lot_b"))
    return {"required": order["quantity"], "lot_a": lot_a, "lot_b": lot_b, "total": lot_a + lot_b}


def score_quantities(output: dict, truth: dict) -> str:
    blocks = QUANTITIES.findall(output["completion"]["text"])
    try:
        recorded = json.loads(blocks[0]) if len(blocks) == 1 else None
    except (ValueError, TypeError):
        recorded = None
    if not isinstance(recorded, dict) or set(recorded) != set(truth) or any(
            type(value) is not int for value in recorded.values()):
        return "numeric_output_unavailable"
    if any(recorded[key] != truth[key] for key in ("required", "lot_a", "lot_b")):
        return "wrong_recorded_quantity"
    if recorded["total"] != truth["total"]:
        return "wrong_sum_given_correct_quantities"
    if output["status"] != "answered":
        return "correct_values_invalid_answer"
    return "correct_values_correct_route" if output["correct"] else "correct_values_wrong_route"


async def pool_replay(case: Case, world: int, history: list[dict], generate: Generate) -> dict:
    truth = pool_numbers(case, world)
    outputs = {}
    for arm in arm_order(POOL_ARMS, case, world):
        subtotal = truth["total"] if arm == "verified_subtotal" else None
        addon = message("user", "Verified subtotal for the two lots, if provided: "
                        + json.dumps(subtotal) + ".\n\n" + POOL_INSTRUCTION)
        output = score_output(await generate(history + [addon]), case.answers[world],
                              allow_answer_period=True)
        outputs[arm] = {"message": addon, **output, "numeric_status": score_quantities(output, truth)}
    return {"reference_quantities": truth, "arms": outputs}


async def run_pair(case: Case, placebo: Case, generate: Generate) -> dict:
    terminals: list[dict] = []
    rejections: list[dict] = []
    base = await evaluate(case, generate, MAX_QUERIES, allow_batch=True, run_oracle=False,
                          terminals=terminals, rejections=rejections, allow_answer_period=True)
    worlds = []
    for terminal in terminals:
        for world in terminal["worlds"]:
            missing = [j for j, query in enumerate(case.oracle) if query not in terminal["queries"]]
            status = terminal["status"]
            correct = status == "answered" and terminal["answer"] == case.answers[world]
            if status != "answered":
                category = status
            elif correct:
                category = "autonomous_correct"
            elif missing:
                category = "wrong_without_full_evidence"
            else:
                category = "wrong_with_full_evidence"
            row = {"world": world, "label": case.answers[world], "status": status,
                   "answer": terminal["answer"], "queries": terminal["queries"],
                   "category": category, "missing": [case.oracle[j] for j in missing]}
            history = terminal["history"]
            if status == "answered" and not correct and missing:
                row["history"] = history
                row["acquisition"] = await acquisition_replay(
                    case, placebo, world, history, missing, generate)
            elif case.family == "pool" and status == "answered" and not missing:
                row["history"] = history
                row["use"] = await pool_replay(case, world, history, generate)
            worlds.append(row)
    return {"case_id": case.id, "family": case.family, "autonomous_cost": base["costs"]["autonomous"],
            "rejections": rejections, "worlds": sorted(worlds, key=lambda row: row["world"])}


def valid_use(arms: dict) -> bool:
    return all(a["status"] == "answered" for a in arms.values())


def summarize(rows: list[dict]) -> dict:
    by_family = {}
    for family in FAMILIES:
        cases = [row for row in rows if row["family"] == family]
        worlds = [world for row in cases for world in row["worlds"]]
        acquisitions = [world["acquisition"] for world in worlds if "acquisition" in world]
        uses = [world["use"]["arms"] for world in worlds if "use" in world]
        valid_uses = [arms for arms in uses if valid_use(arms)]
        by_family[family] = {
            "pairs": len(cases), "world_categories": counts(w["category"] for w in worlds),
            "autonomous_rejection_reasons": counts(event["reason"]
                                                   for row in cases for event in row["rejections"]),
            "acquisition_attempted_worlds": len(acquisitions),
            "acquisition_all_three_valid_worlds": sum(a["all_valid"] for a in acquisitions),
            "acquisition_relevant_only_worlds": sum(a["relevant_only"] for a in acquisitions),
            "acquisition_relevant_only_pairs": sum(any(
                w.get("acquisition", {}).get("relevant_only", False) for w in row["worlds"])
                for row in cases),
            "acquisition_by_arm": {
                arm: {"valid": sum(a["arms"][arm]["status"] == "answered" for a in acquisitions),
                      "correct": sum(a["arms"][arm]["correct"] for a in acquisitions)}
                for arm in ACQUISITION_ARMS
            },
            "use_attempted_worlds": len(uses), "use_both_valid_worlds": len(valid_uses),
            "use_pairs_with_two_eligible_worlds": sum(
                all("use" in w and valid_use(w["use"]["arms"]) for w in row["worlds"])
                for row in cases),
            "use_both_worlds_correct_by_arm": {
                arm: sum(all("use" in w and valid_use(w["use"]["arms"])
                             and w["use"]["arms"][arm]["correct"] for w in row["worlds"])
                         for row in cases) for arm in POOL_ARMS
            },
            "use_by_arm": {arm: {"valid": sum(a[arm]["status"] == "answered" for a in uses),
                                 "correct": sum(a[arm]["correct"] for a in uses),
                                 "numeric_statuses": counts(a[arm]["numeric_status"] for a in uses)}
                           for arm in POOL_ARMS},
            "use_subtotal_wins": sum(not a["unassisted"]["correct"] and
                                     a["verified_subtotal"]["correct"] for a in valid_uses),
            "use_subtotal_losses": sum(a["unassisted"]["correct"] and
                                       not a["verified_subtotal"]["correct"] for a in valid_uses),
        }
    return by_family


async def run(args: argparse.Namespace) -> dict:
    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = Renderer(tokenizer, TOOLS)
    repo = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    cases = [(linked_case(family, i, seed=SEED), linked_case(family, i, seed=SEED, placebo=True))
             for family in FAMILIES for i in range(PAIRS_PER_FAMILY)]
    rows = []
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        async def generate(history: list[dict]):
            return await request_completion(client, renderer, args.endpoint, args.model,
                                            MAX_TOKENS, history)
        for start in range(0, len(cases), args.concurrency):
            chunk = cases[start:start + args.concurrency]
            rows.extend(await asyncio.gather(*(run_pair(case, filler, generate)
                                               for case, filler in chunk)))
            print(f"Completed {len(rows)}/{len(cases)} pairs", flush=True)
    final = {
        "config": {**vars(args), "protocol": "rsi_terminal_probe_v1", "revision": revision,
                   "seed": SEED, "temperature": 0, "max_queries": MAX_QUERIES,
                   "max_tokens": MAX_TOKENS, "pairs_per_family": PAIRS_PER_FAMILY},
        "total_pairs": len(rows), "by_family": summarize(rows),
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rsi_terminal_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp = output / "rsi_terminal_summary.tmp"
    temp.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output / "rsi_terminal_summary.json")
    return final


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
