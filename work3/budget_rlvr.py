"""Offline pilot for using verified short answers to estimate long-rollout rewards."""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep

from work3.budget_rlvr_metrics import (FOLDS, LONG, PREFIX, PROTOCOL, QUESTIONS, REPEATS, SHORT,
                                     compare, exact_panel, observed)
from work3.continual_rlvr import identify_inputs
from work3.continual_rlvr_data import (PARQUET_SHA, PROMPT_BOUND, REVISIONS, canonical, digest, load_bundle,
                                     load_prompt_tokenizer, normalize, prompt_ids, rational, sha256_file,
                                     tokenizer_fingerprint)
from work3.wiki_rsi_progress import REPO, Progress, atomic_json, progress_root

OLD_BUNDLE = "6c2c589d4e69f8272d52e185bcee4ad403d971388b7b8cb7aa6a45ae991c619f"
PANEL_SHA = "d8585b1a5389f98d680728d85a69b2e25e9d9a1340b667bf7eebb7e9cd4bbb8f"
MODEL_WEIGHTS = {
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
}
GPUS = 4
BATCH = 16
SEED = 20261124
PHASE = "budget_rlvr"
LIMITS = {"smoke": 600, "pilot": 7200}


def selection(parquet, exclude, tokenizer):
    from pyarrow.parquet import ParquetFile

    groups = defaultdict(list)
    for batch in ParquetFile(parquet).iter_batches():
        for raw in batch.to_pylist():
            case = normalize("a", raw)
            if case and case["group"] not in exclude:
                groups[case["group"]].append(case)
    cases = []
    for variants in groups.values():
        selected = min(variants, key=lambda row: (row["id"], row["target"]))
        if any(row["id"] == selected["id"] and row["target"] != selected["target"] for row in variants):
            continue
        cases.append(selected)
    cases.sort(key=lambda case: digest(PROTOCOL + ":" + case["id"]))
    result = []
    for case in cases:
        if 1 <= len(prompt_ids(tokenizer, case)) <= PROMPT_BOUND:
            result.append(case)
        if len(result) == QUESTIONS:
            return result
    raise ValueError("fewer than 256 disjoint, token-eligible rational math questions")


def old_exclusion(path):
    if sha256_file(path) != OLD_BUNDLE:
        raise ValueError("expected the original C1 bundle_r2.json")
    splits = load_bundle(path)["splits"]
    return {case["group"] for name in ("a_train", "a_dev") for case in splits[name]}


def prepare(args):
    if args.math is None or sha256_file(args.math) != PARQUET_SHA["a"]:
        raise ValueError("prepare requires the original RLVR-MATH train parquet")
    exclude = old_exclusion(args.bundle)
    cases = selection(args.math, exclude, load_prompt_tokenizer(args.model))
    panel = {"protocol": PROTOCOL, "math_revision": REVISIONS["a"], "math_sha256": PARQUET_SHA["a"],
             "excluded_bundle_sha256": OLD_BUNDLE, "tokenizer_sha256": tokenizer_fingerprint(args.model),
             "cases": cases}
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    if args.panel.exists():
        if json.loads(args.panel.read_text(encoding="utf-8")) != panel:
            raise ValueError("refusing to overwrite a different pilot panel")
    else:
        atomic_json(args.panel, panel)
    return {"questions": len(cases), "panel_sha256": sha256_file(args.panel)}


def load_panel(path):
    panel = json.loads(path.read_text(encoding="utf-8"))
    if (panel["protocol"], panel["math_revision"], panel["math_sha256"], panel["excluded_bundle_sha256"]) != (
            PROTOCOL, REVISIONS["a"], PARQUET_SHA["a"], OLD_BUNDLE):
        raise ValueError("wrong pilot panel provenance")
    cases = panel["cases"]
    if (len(cases) != QUESTIONS or len({case["id"] for case in cases}) != QUESTIONS
            or len({case["group"] for case in cases}) != QUESTIONS
            or any(case["id"] != digest(canonical(case["prompt"])) or case["task"] != "a"
                   or rational(case["target"]) is None for case in cases)):
        raise ValueError("invalid pilot question panel")
    return panel


def preflight(args):
    exclude = old_exclusion(args.bundle)
    panel = load_panel(args.panel)
    if sha256_file(args.panel) != PANEL_SHA:
        raise ValueError("pilot panel differs from the frozen 256-question input")
    source, _ = identify_inputs(args)
    if any(source["model_sha256"].get(name) != checksum for name, checksum in MODEL_WEIGHTS.items()):
        raise ValueError("expected the exact frozen C1 Qwen2.5-3B-Instruct base checkpoint")
    if panel["tokenizer_sha256"] != tokenizer_fingerprint(args.model):
        raise ValueError("pilot panel tokenizer differs from the local model")
    if any(case["group"] in exclude for case in panel["cases"]):
        raise ValueError("pilot math panel overlaps the original C1 math data")
    tokenizer = load_prompt_tokenizer(args.model)
    maximum = max(len(prompt_ids(tokenizer, case)) for case in panel["cases"])
    if not 1 <= maximum <= PROMPT_BOUND:
        raise ValueError("pilot panel prompt exceeds the 768-token bound")
    return {**source, "protocol": PROTOCOL, "panel_sha256": PANEL_SHA, "questions": QUESTIONS,
            "samples_per_question": REPEATS, "folds": FOLDS, "prefix_cap": PREFIX, "short_cap": SHORT,
            "long_cap": LONG, "seed": SEED, "batch_size": BATCH, "max_prompt_tokens": maximum}


def cohort(panel, phase, original_bundle):
    if phase == "smoke":
        return load_bundle(original_bundle)["splits"]["a_dev"][:8], 2
    return panel["cases"], REPEATS


def worker(args):
    from work3.budget_rlvr_sampler import Sampler

    rank = args.rank
    cases, repeats = cohort(load_panel(args.panel), args.kind, args.bundle)
    jobs = [(case, repeat) for case in cases[rank::GPUS] for repeat in range(repeats)]
    sampler = Sampler.load(args.model, rank)
    print(f"rank {rank}: {len(jobs)} samples on {sampler.device}", flush=True)
    counts = {"done": 0, "long": 0, "tokens": 0}
    with (args.directory / f"rank-{rank}.jsonl").open("x", encoding="utf-8") as stream:
        for offset in range(0, len(jobs), BATCH):
            batch = jobs[offset:offset + BATCH]
            rows = sampler.branches(batch, SEED + 500009 * rank + 104729 * (offset // BATCH))
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["done"] += 1
                counts["long"] += not row["prefix_ended"]
                counts["tokens"] += row["base_tokens"] + row["short_tokens"] + row["long_tokens"]
            stream.flush()
            atomic_json(args.directory / f"rank-{rank}-status.json", counts)


def status(directory):
    totals = dict(done=0, long=0, tokens=0)
    for rank in range(GPUS):
        path = directory / f"rank-{rank}-status.json"
        if path.is_file():
            counts = json.loads(path.read_text(encoding="utf-8"))
            for key in totals:
                totals[key] += counts[key]
    return totals


def stop_workers(workers):
    for process in workers:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    for process in workers:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def publish_progress(progress, counts, total, detail, state="running"):
    progress.update(state, counts["done"], total, detail=detail,
                    metrics={"active_gpus": GPUS if state == "running" else 0,
                             "rollout_tokens": counts["tokens"], "long_branches": counts["long"]})


def sample(args, directory, progress, total, deadline):
    workers = []
    try:
        for rank in range(GPUS):
            command = [sys.executable, "-m", "work3.budget_rlvr", "worker", "--model", str(args.model.resolve()),
                       "--panel", str(args.panel.resolve()), "--bundle", str(args.bundle.resolve()),
                       "--kind", args.phase, "--rank", str(rank),
                       "--directory", str(directory)]
            with (directory / f"rank-{rank}.log").open("wb") as log:
                workers.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True))
        while True:
            codes = [process.poll() for process in workers]
            publish_progress(progress, status(directory), total, "budget_rlvr_sampling")
            if any(code not in {0, None} for code in codes):
                raise RuntimeError(f"worker exited early: {codes}; see rank logs")
            if all(code == 0 for code in codes):
                return
            if monotonic() >= deadline:
                raise TimeoutError("budget RLVR reached the frozen wall-clock limit")
            sleep(2)
    finally:
        stop_workers(workers)


def stop(signum, _frame):
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def check_smoke(root, run_id, inputs):
    path = root / run_id / "wiki_rsi_budget_rlvr_summary.json"
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if (smoke.get("phase") != "smoke" or smoke.get("verdict") != "PASS"
            or smoke.get("provenance") != inputs or smoke.get("samples") != 16
            or not smoke.get("truncated_at_prefix")):
        raise ValueError("pilot requires a PASS smoke with identical model, panel and runtime")


def run(args):
    if os.name != "posix":
        raise RuntimeError("run the GPU pilot on the factory Linux environment")
    started = monotonic()
    deadline = started + LIMITS[args.phase]
    directory = progress_root() / args.id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    progress = Progress(directory, PHASE)
    signal.signal(signal.SIGTERM, stop)
    total = 0
    try:
        with progress.heartbeat():
            progress.update("running", detail="budget_rlvr_preflight")
            inputs = preflight(args)
            atomic_json(directory / "inputs.json", inputs)
            if args.phase == "pilot":
                check_smoke(directory.parent, args.smoke_run, inputs)
            cases, repeats = cohort(load_panel(args.panel), args.phase, args.bundle)
            total = len(cases) * repeats
            if monotonic() >= deadline:
                raise TimeoutError("input checks consumed the wall-clock limit")
            sample(args, directory, progress, total, deadline)
            rows = []
            for rank in range(GPUS):
                with (directory / f"rank-{rank}.jsonl").open(encoding="utf-8") as stream:
                    rows.extend(json.loads(line) for line in stream)
            exact_panel(rows, cases, repeats)
            publish_progress(progress, status(directory), total, "budget_rlvr_comparing")
            if args.phase == "smoke":
                result = observed(rows)
                if result["truncated_at_prefix"] == 0:
                    raise RuntimeError("smoke completed no actual short/long fork")
                result["verdict"] = "PASS"
            else:
                result = compare(rows, cases)
            if monotonic() >= deadline:
                raise TimeoutError("comparison reached the frozen wall-clock limit")
            result.update(provenance=inputs, phase=args.phase, wall_seconds=monotonic() - started,
                          panel_record_sha256={f"rank-{rank}.jsonl": sha256_file(directory / f"rank-{rank}.jsonl")
                                               for rank in range(GPUS)})
            atomic_json(directory / "wiki_rsi_budget_rlvr_summary.json", result)
            state = "blocked" if result["verdict"] == "NO_GO" else "done"
            publish_progress(progress, status(directory), total, "budget_rlvr_" + result["verdict"].lower(), state)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        counts = status(directory)
        atomic_json(directory / "incomplete.json", {"verdict": "INCOMPLETE", "phase": args.phase,
                    "repo_revision": progress.commit, "error": str(exc), "completed_samples": counts["done"],
                    "planned_samples": total})
        publish_progress(progress, counts, total, "budget_rlvr_incomplete", "failed")
        raise


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "preflight", "smoke", "pilot", "worker"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, help="original continual RLVR bundle_r2.json")
    parser.add_argument("--math", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--smoke-run")
    parser.add_argument("--rank", type=int, choices=range(GPUS), help=argparse.SUPPRESS)
    parser.add_argument("--kind", choices=LIMITS, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "worker":
        worker(args)
        return
    if args.bundle is None:
        parser.error("--bundle is required")
    if args.phase in LIMITS:
        if not re.fullmatch(r"budget-rlvr-[a-z0-9][a-z0-9-]{0,55}", args.id or ""):
            parser.error("--id must be a fresh budget-rlvr-<name>")
        if args.phase == "pilot" and not re.fullmatch(r"budget-rlvr-[a-z0-9][a-z0-9-]{0,55}", args.smoke_run or ""):
            parser.error("pilot requires --smoke-run budget-rlvr-<name>")
        run(args)
    else:
        result = prepare(args) if args.phase == "prepare" else preflight(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
