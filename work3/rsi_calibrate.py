"""Calibrate evidence presentation on the frozen paired RSI inventory tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from work3.rsi_probe import (
    ORACLE_FINISH, TOOLS, Completion, Generate, Renderer, counts, decide, evaluate,
    evidence_history, initial_history, message, request_completion, tool_message,
)
from work3.rsi_tasks import Case, SPLITS, make_case, make_cases

DIAGNOSTIC_SEED = 20261117
CONFIRMATION_SEED = 20261118
CONFIRMATION_CASES = 30
MAX_QUERIES = 4
MAX_TOKENS = 256
GATE = {"valid_world_rate": 0.95, "pair_accuracy": 0.70, "label_b_accuracy": 0.80}
CONDITIONS = ("tool_direct", "tool_compare", "inline_direct", "inline_compare")
COMPARE = (
    "The lookup phase is complete. Use the records above to decide the route. "
    "First write down the relevant quantities from the records, apply the rule stated in the "
    "question, and show the resulting comparisons. Then end with <answer>A</answer> or "
    "<answer>B</answer>. Make no further calls."
)
# Fixed synthetic case indices at the published seed, stratified using the v2 run.
DIAGNOSTIC_GROUPS = (
    ("bundle", False, "prior_format_error", (0, 4, 6, 7, 8, 10)),
    ("bundle", False, "prior_autonomous_solved", (5, 9, 11)),
    ("bundle", False, "prior_post_observation_error", (1, 2, 3)),
    ("reserved", False, "raw_stock_insufficient", (0, 24, 26, 45, 56, 59)),
    ("reserved", False, "raw_stock_sufficient", (1, 2, 3, 4, 5, 6)),
    ("bundle", True, "prior_control_format_error", (1, 2)),
    ("bundle", True, "prior_control_solved", (0, 5)),
    ("reserved", True, "override_control", (0, 1, 2, 3)),
)


def diagnostic_cases() -> list[tuple[Case, str]]:
    return [(make_case(family, DIAGNOSTIC_SEED, index, control), group)
            for family, control, group, indices in DIAGNOSTIC_GROUPS for index in indices]


def oracle_prompt(case: Case, world: dict, condition: str) -> list[dict]:
    presentation, instruction_type = condition.split("_")
    instruction = ORACLE_FINISH if instruction_type == "direct" else COMPARE
    if presentation == "tool":
        return evidence_history(case, world, instruction)
    evidence = ["Raw lookup evidence, in lookup order:"]
    for query in case.oracle:
        evidence.append(query + "\n" + tool_message(world, query)["content"])
    return initial_history(case) + [message("user", "\n\n".join(evidence + [instruction]))]


def score_output(response: Completion, label: str, *, allow_answer_period: bool = False) -> dict:
    action = decide(response, allow_answer_period=allow_answer_period)
    return {
        "answer": action.answer,
        "correct": action.answer == label,
        "status": "unexpected_tool" if action.status == "tool" else action.status,
        "rejection": action.rejection,
        "completion": asdict(response),
    }


async def oracle_arm(case: Case, generate: Generate, condition: str) -> list[dict]:
    return [score_output(await generate(oracle_prompt(case, world, condition)), case.answers[i])
            for i, world in enumerate(case.worlds)]


async def run_case(case: Case, group: str, generate: Generate, condition: str | None) -> dict:
    row = {"case_id": case.id, "family": case.family, "group": group,
           "control": case.control, "labels": case.answers}
    if condition is not None:
        return {**row, "conditions": {condition: await oracle_arm(case, generate, condition)}}

    rejections: list[dict] = []
    baseline_outputs: list[Completion] = []
    baseline = await evaluate(case, generate, MAX_QUERIES, rejections=rejections,
                              oracle_completions=baseline_outputs)
    row["autonomous"] = {
        "branches": baseline["autonomous"],
        "correct": baseline["autonomous_correct"],
        "sufficient_evidence": baseline["autonomous_sufficient_evidence"],
        "fork_at_query": baseline["fork_at_query"],
        "diagnosis": baseline["diagnosis"],
        "costs": baseline["costs"]["autonomous"],
        "rejections": rejections,
    }
    row["conditions"] = {}
    if not case.control:
        row["conditions"]["tool_direct"] = [
            score_output(response, label) for response, label in zip(baseline_outputs, case.answers)
        ]
        for variant in CONDITIONS[1:]:
            row["conditions"][variant] = await oracle_arm(case, generate, variant)
    return row


def oracle_metrics(rows: list[dict], condition: str) -> dict:
    labels = {"A": {"worlds": 0, "correct": 0}, "B": {"worlds": 0, "correct": 0}}
    valid, pairs_correct, same_answer, wins, losses = 0, 0, 0, 0, 0
    statuses = []
    prompt_tokens, completion_tokens = 0, 0
    baseline_available = condition != "tool_direct" and all(
        "tool_direct" in row["conditions"] for row in rows)
    for row in rows:
        outputs = row["conditions"][condition]
        solved = all(output["correct"] for output in outputs)
        pairs_correct += solved
        same_answer += outputs[0]["answer"] is not None and outputs[0]["answer"] == outputs[1]["answer"]
        if baseline_available:
            baseline_solved = all(output["correct"] for output in row["conditions"]["tool_direct"])
            wins += solved and not baseline_solved
            losses += baseline_solved and not solved
        for label, output in zip(row["labels"], outputs):
            labels[label]["worlds"] += 1
            labels[label]["correct"] += output["correct"]
            statuses.append(output["status"])
            valid += output["status"] == "answered"
            prompt_tokens += output["completion"]["prompt_tokens"]
            completion_tokens += output["completion"]["completion_tokens"]
    return {
        "pairs": len(rows), "pairs_correct": pairs_correct, "worlds_with_answer": valid,
        "same_answer_pairs": same_answer, "by_label": labels, "statuses": counts(statuses),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "paired_vs_tool_direct": {"wins": wins, "losses": losses} if baseline_available else None,
    }


def autonomous_metrics(rows: list[dict]) -> dict:
    states = [row["autonomous"] for row in rows]
    rejects = [event for state in states for event in state["rejections"]]
    return {
        "pairs": len(rows),
        "pairs_correct": sum(all(state["correct"]) for state in states),
        "pairs_with_tool": sum(any(branch["queries"] for branch in state["branches"])
                               for state in states),
        "statuses": counts(branch["status"] for state in states for branch in state["branches"]),
        "diagnoses": counts(state["diagnosis"] for state in states),
        "rejected_generations": len(rejects),
        "rejection_reasons": counts(event["reason"] for event in rejects),
        "rejection_after_executed_queries": counts(event["executed_queries"] for event in rejects),
    }


def summarize(rows: list[dict], condition: str | None) -> dict:
    targets = [row for row in rows if not row["control"]]
    by_family = {}
    confirmation_checks = {}
    for family in SPLITS["heldout"]:
        family_rows = [row for row in targets if row["family"] == family]
        arms = CONDITIONS if condition is None else (condition,)
        metrics = {arm: oracle_metrics(family_rows, arm) for arm in arms}
        by_family[family] = {"conditions": metrics}
        if condition is None:
            controls = [row for row in rows if row["control"] and row["family"] == family]
            by_family[family]["autonomous"] = autonomous_metrics(family_rows)
            by_family[family]["controls"] = autonomous_metrics(controls)
        else:
            chosen = metrics[condition]
            actual = {
                "valid_world_rate": chosen["worlds_with_answer"] / (2 * chosen["pairs"]),
                "pair_accuracy": chosen["pairs_correct"] / chosen["pairs"],
                "label_b_accuracy": chosen["by_label"]["B"]["correct"] / chosen["by_label"]["B"]["worlds"],
            }
            confirmation_checks[family] = {
                "observed": actual, "pass": all(actual[key] >= limit for key, limit in GATE.items()),
            }
    result = {"total_pairs": len(rows), "target_pairs": len(targets),
              "control_pairs": len(rows) - len(targets), "by_family": by_family}
    if condition is not None:
        result["confirmation"] = {
            "thresholds": GATE, "by_family": confirmation_checks,
            "pass": all(check["pass"] for check in confirmation_checks.values()),
        }
    return result


async def run(args: argparse.Namespace) -> dict:
    import httpx
    from transformers import AutoTokenizer

    if args.phase == "diagnostic":
        seed = DIAGNOSTIC_SEED
        cases = diagnostic_cases()
    else:
        seed = CONFIRMATION_SEED
        cases = [(case, "fresh_confirmation")
                 for case in make_cases("heldout", CONFIRMATION_CASES, 0, seed)]
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    renderer = Renderer(tokenizer, TOOLS)
    rows = []
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        async def generate(history: list[dict]) -> Completion:
            return await request_completion(client, renderer, args.endpoint, args.model, MAX_TOKENS, history)

        for start in range(0, len(cases), args.concurrency):
            batch = cases[start:start + args.concurrency]
            rows.extend(await asyncio.gather(
                *(run_case(case, group, generate, args.condition) for case, group in batch)))
            print(f"Completed {len(rows)}/{len(cases)} pairs", flush=True)

    result = {
        "config": {
            **vars(args), "revision": revision, "protocol": "rsi_calibration_v1", "seed": seed,
            "temperature": 0, "max_queries": MAX_QUERIES, "max_tokens": MAX_TOKENS,
            "python": platform.python_version(), "transformers": version("transformers"),
            "httpx": version("httpx"),
        },
        **summarize(rows, args.condition),
    }
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rsi_calibration_cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = output / "rsi_calibration_summary.json"
    temporary = summary.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnostic", "confirmation"), default="diagnostic")
    parser.add_argument("--condition", choices=CONDITIONS, help="Frozen condition for confirmation")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8187")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer path")
    parser.add_argument("--out-dir", required=True, help="New directory for this run")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if (args.phase == "confirmation") != (args.condition is not None):
        parser.error("Supply --condition only with --phase confirmation, where it is required.")
    if args.concurrency < 1 or Path(args.out_dir).exists():
        parser.error("Use positive concurrency and a new output directory.")
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
