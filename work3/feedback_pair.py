"""Offline crossed feedback pilot. Use the factory guide for the normal entry points."""

import argparse
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep, time

from work3.budget_rlvr import stop, stop_workers
from work3.continual_rlvr_data import load_prompt_tokenizer, sha256_file, tokenizer_fingerprint
from work3.feedback_pair_data import (DATA_REVISION, DRAW_CAP, PROTOCOL, TARGET, Verifier, check_references,
                                      clean_code, read_rows)
from work3.feedback_pair_metrics import (BRANCHES, OUTPUT_CAP, REPEATS, TOKEN_CAP, collect_scored, job_seed,
                                         order_key, panel, prompt)
from work3.wiki_rsi_progress import REPO, Progress, atomic_json, progress_root


GPUS = 4
BATCH = 16
JUDGE_THREADS = 4  # per rank
SMOKE_TASKS = 32
LIMITS = {"smoke": 1800, "pilot": 43200}
RUN_ID = re.compile(r"pair-repair-[a-z0-9][a-z0-9-]{0,48}")
SUMMARY = "wiki_rsi_pair_repair_summary.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(args, directory, progress):
    progress.update("running", detail="pair_source")
    verifier = Verifier()
    verifier.probe()
    tasks, hashes = read_rows(args.source)

    def report(done, total, ready):
        progress.update("running", done, total, detail="pair_references", metrics={"verified_pairs": ready})

    splits, checked = check_references(tasks, verifier, args.cpu_jobs, report)
    splits = {name: sorted(splits.get(name, []), key=lambda task: order_key(task["id"])) for name in ("train", "sealed")}
    counts = {name: len(rows) for name, rows in splits.items()}
    provenance = {"protocol": PROTOCOL, "repo_revision": progress.commit, "dataset_revision": DATA_REVISION,
                  "shards": hashes, "verifier": verifier.version()}
    payload = {"provenance": provenance, "tasks": splits}
    atomic_json(directory / "manifest.json", payload)
    return {"verdict": "READY" if min(counts.values()) >= TARGET and len(splits["train"]) >= SMOKE_TASKS
            else "DATA-NO-GO", "structural_tasks": len(tasks), "reference_checks": checked, "ready_by_split": counts,
            "manifest_sha256": sha256_file(directory / "manifest.json"), "provenance": provenance}


def model_input(args, directory, progress):
    manifest_file = args.manifest.resolve()
    manifest = load(manifest_file)
    prepared = load(manifest_file.parent / SUMMARY)
    checksum = sha256_file(manifest_file)
    if (manifest["provenance"]["protocol"] != PROTOCOL or prepared["phase"] != "prepare"
            or prepared["verdict"] != "READY" or prepared["manifest_sha256"] != checksum
            or manifest["provenance"]["repo_revision"] != progress.commit):
        raise ValueError("requires READY data prepared by this commit with the exact manifest hash")
    splits = manifest["tasks"]
    ids = [task["id"] for rows in splits.values() for task in rows]
    if (set(splits) != {"train", "sealed"} or len(set(ids)) != len(ids)
            or any(not TARGET <= len(rows) <= DRAW_CAP or any(row["split"] != name for row in rows)
                   for name, rows in splits.items())
            or {row["contest"] for row in splits["train"]} & {row["contest"] for row in splits["sealed"]}):
        raise ValueError("invalid or overlapping task splits")
    verifier = Verifier()
    verifier.probe()
    if verifier.version() != manifest["provenance"]["verifier"]:
        raise ValueError("execution environment changed after data preparation")
    model = args.model.resolve()
    model_files = sorted(model.glob("model*.safetensors"))
    if not model_files or any(not path.is_file() for path in (model / "config.json", model / "generation_config.json")):
        raise ValueError("expected a complete local safetensors model with config and generation config")
    tokenizer = load_prompt_tokenizer(model)
    task_counts = {name: len(rows) for name, rows in splits.items()}
    splits = {name: [task for task in rows if 0 < len(prompt(tokenizer, task)) <= TOKEN_CAP - OUTPUT_CAP]
              for name, rows in splits.items()}
    inputs = {"manifest_sha256": checksum, "repo_revision": progress.commit, "protocol": PROTOCOL,
              "model_sha256": {path.name: sha256_file(path) for path in model_files + sorted(model.glob("*.json"))},
              "tokenizer_sha256": tokenizer_fingerprint(model), "verifier": verifier.version(),
              "versions": {package: importlib.metadata.version(package) for package in
                           ("torch", "transformers", "numpy", "tokenizers", "safetensors")},
              "python": sys.version.split()[0], "target": TARGET, "repeats": REPEATS, "gpus": GPUS,
              "batch": BATCH, "token_cap": TOKEN_CAP, "output_cap": OUTPUT_CAP,
              "initial_prompt_exclusions": {name: task_counts[name] - len(rows) for name, rows in splits.items()}}
    atomic_json(directory / "inputs.json", inputs)
    return inputs, splits


def batches(jobs):
    """Batch similar lengths while retaining an exact per-request context cap."""
    jobs = sorted(jobs, key=lambda job: len(job[1]))
    for offset in range(0, len(jobs), BATCH):
        yield jobs[offset:offset + BATCH]


def decode(model, tokenizer, device, ids, seed):
    import torch
    from transformers import GenerationConfig

    width = max(map(len, ids))
    if width > TOKEN_CAP - OUTPUT_CAP:
        raise ValueError("prompt exceeds frozen prefill cap")
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    pad = tokenizer.pad_token_id
    if pad is None or None in eos_ids:
        raise ValueError("tokenizer requires PAD and EOS")
    torch.manual_seed(seed)
    tensor = torch.tensor([[pad] * (width - len(row)) + list(row) for row in ids], device=device)
    mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids], device=device)
    started = monotonic()
    with torch.inference_mode():
        output = model.generate(input_ids=tensor, attention_mask=mask, generation_config=GenerationConfig(
            do_sample=True, temperature=0.7, top_p=0.95, top_k=0, max_new_tokens=OUTPUT_CAP,
            eos_token_id=eos, pad_token_id=pad, use_cache=True))[:, width:].tolist()
    duration = (monotonic() - started) / len(ids)
    rows = []
    for given, generated in zip(ids, output, strict=True):
        end = next((i + 1 for i, token in enumerate(generated) if token in eos_ids), None)
        response = clean_code(tokenizer.decode(generated[:end], skip_special_tokens=True)) if end else None
        rows.append((response, len(given), end or len(generated), duration, end is None))
    return rows


def initial_row(task, reply, tokenizer, verifier):
    code, prefills, output, seconds, truncated = reply
    feedback, feedback_calls, feedback_seconds = verifier.evaluate(code, task["feedback"])
    scores, score_calls, score_seconds = verifier.evaluate(code, task["score"])
    scores = [status == "pass" for status, _ in scores]
    observed_failure = any(status != "pass" for status, _ in feedback)
    chosen = None if not observed_failure or code is None else panel(task, code, feedback, tokenizer)
    return {"id": task["id"], "contest": task["contest"], "split": task["split"], "initial": code,
            "observed_failure": observed_failure, "initial_correct": all(scores), "initial_pass_mask": scores,
            "eligible": chosen is not None,
            "initial_tokens": prefills + output, "initial_output_tokens": output, "initial_truncated": truncated,
            "initial_executions": feedback_calls + score_calls, "initial_generation_seconds": seconds,
            "initial_judge_seconds": feedback_seconds + score_seconds, **(chosen or {})}


def repaired_row(task, initial, reply, verifier):
    code, prefills, output, seconds, truncated = reply
    outcomes, executions, judge_seconds = verifier.evaluate(code, task["score"])
    outcomes = [status == "pass" for status, _ in outcomes]
    prior = initial["initial_pass_mask"]
    return {"success": all(outcomes), "prompt_tokens": prefills, "output_tokens": output,
            "tokens": prefills + output, "generation_seconds": seconds, "judge_seconds": judge_seconds,
            "executions": executions, "truncated": truncated, "heldout_pass_mask": outcomes,
            "retained": sum(a and b for a, b in zip(prior, outcomes)) / sum(prior) if any(prior) else None,
            "repair_fraction": sum(not a and b for a, b in zip(prior, outcomes)) / prior.count(False)
            if not all(prior) else None}


def status_path(directory, stage, rank):
    return directory / f"{stage}-{rank}-status.json"


def rank_worker(args):
    import torch
    from transformers import AutoModelForCausalLM

    if torch.cuda.device_count() != GPUS:
        raise RuntimeError("exactly four visible GPUs are required")
    rank = args.rank
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    directory, stage = args.directory, args.stage
    device = torch.device("cuda", rank)
    verifier = Verifier()
    verifier.probe()
    tokenizer = load_prompt_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                torch_dtype=torch.bfloat16, use_safetensors=True).to(device).eval()
    assignment = load(directory / f"{stage}-plan.json")
    initial = stage == "initial"
    jobs = []
    if initial:
        for task in assignment[rank::GPUS]:
            ids = prompt(tokenizer, task)
            if len(ids) > TOKEN_CAP - OUTPUT_CAP:
                raise ValueError(f"initial prompt exceeds configured bound: {task['id']}")
            jobs.append(((task, None, None, 0), ids))
    else:
        for index, (task, state) in enumerate(assignment):
            arms = BRANCHES if state["eligible"] else ("regenerate",)
            for branch, arm in enumerate(arms):
                examples = state.get("examples", {}).get(arm, ())
                for repeat in range(REPEATS):
                    if (index + branch * REPEATS + repeat) % GPUS == rank:
                        ids = prompt(tokenizer, task, state["initial"] if examples else None, examples)
                        jobs.append(((task, state, arm, repeat), ids))

    def score(metadata, response):
        task, state, arm, repeat = metadata
        if initial:
            return initial_row(task, response, tokenizer, verifier)
        return {"id": task["id"], "arm": arm, "repeat": repeat, **repaired_row(task, state, response, verifier)}

    done, tokens = 0, 0
    with (directory / f"{stage}-{rank}.jsonl").open("x", encoding="utf-8") as stream, \
            ThreadPoolExecutor(max_workers=JUDGE_THREADS) as threads:
        for batch_index, batch in enumerate(batches(jobs)):
            metadata, prompts = zip(*batch)
            responses = decode(model, tokenizer, device, prompts, job_seed(stage, rank, batch_index))
            for row in threads.map(score, metadata, responses):
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                tokens += row["initial_tokens" if initial else "tokens"]
                done += 1
            stream.flush()
            atomic_json(status_path(directory, stage, rank), {"done": done, "total": len(jobs), "rollout_tokens": tokens,
                                                              "updated_at": int(time())})


def status(directory, stage, total, state, progress, detail=None):
    workers, tokens, done = [], 0, 0
    for rank in range(GPUS):
        path = status_path(directory, stage, rank)
        if path.is_file():
            row = load(path)
            workers.append({"name": f"rank-{rank}", "device": str(rank), "detail": f"pair_{stage}", **row})
            tokens += row["rollout_tokens"]
            done += row["done"]
    progress.update(state, done, total, detail=detail or f"pair_{stage}", workers=workers,
                    metrics={"active_gpus": GPUS if state == "running" else 0, "rollout_tokens": tokens})


def sample(args, directory, progress, stage, plan, total, deadline):
    atomic_json(directory / f"{stage}-plan.json", plan)
    workers = []
    try:
        for rank in range(GPUS):
            command = [sys.executable, "-m", "work3.feedback_pair", "worker", "--stage", stage, "--rank", str(rank),
                       "--directory", str(directory), "--model", str(args.model.resolve())]
            with (directory / f"{stage}-{rank}.log").open("wb") as log:
                workers.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True))
        while True:
            codes = [process.poll() for process in workers]
            status(directory, stage, total, "running", progress)
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"{stage} worker exited: {codes}; inspect rank logs")
            if monotonic() >= deadline:
                raise TimeoutError("wall-clock limit reached")
            if all(code == 0 for code in codes):
                break
            sleep(2)
    finally:
        stop_workers(workers)
    rows = []
    for rank in range(GPUS):
        with (directory / f"{stage}-{rank}.jsonl").open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream)
    if len(rows) != total:
        raise ValueError(f"{stage} emitted {len(rows)} records; expected {total}")
    return rows


def plan_repairs(tasks, rows):
    states = {row["id"]: row for row in rows}
    if len(states) != len(rows) or set(states) != {task["id"] for task in tasks}:
        raise ValueError("initial panel has duplicate or missing tasks")
    counts = {name: Counter() for name in ("train", "sealed")}
    repairs = []
    for task in tasks:
        row = states[task["id"]]
        count = counts[task["split"]]
        count["drawn"] += 1
        count["initial_correct"] += row["initial_correct"]
        if not row["observed_failure"]:
            continue
        count["initial_failures"] += 1
        count["invalid_initial"] += row["initial"] is None
        if row["eligible"]:
            count["eligible"] += 1
            repairs.append((task, row))
        elif task["split"] == "sealed":
            repairs.append((task, row))
    return repairs, {name: dict(count) for name, count in counts.items()}


def join(states, records):
    by_key = {(row["id"], row["arm"], row["repeat"]): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("duplicate repair branch")
    result = {"train": [], "sealed": []}
    for _, state in states:
        arms = BRANCHES if state["eligible"] else ("regenerate",)
        branches = {arm: [by_key.pop((state["id"], arm, repeat)) for repeat in range(REPEATS)] for arm in arms}
        result[state["split"]].append({**state, "branches": branches})
    if by_key:
        raise ValueError("unexpected repair branch")
    return result


def experiment(args, directory, progress, started):
    deadline = started + LIMITS[args.phase]
    progress.update("running", detail="pair_preflight")
    inputs, splits = model_input(args, directory, progress)
    if any(len(rows) < TARGET for rows in splits.values()):
        return {"verdict": "DATA-NO-GO", "provenance": inputs,
                "prompt_eligible": {name: len(rows) for name, rows in splits.items()}}
    if args.phase == "pilot":
        smoke = load(directory.parent / args.smoke_run / SUMMARY)
        if (smoke["phase"], smoke["verdict"], smoke["provenance"]) != ("smoke", "PASS", inputs):
            raise ValueError("pilot requires a PASS four-card smoke with identical provenance")
        tasks = splits["train"] + splits["sealed"]
    else:
        tasks = splits["train"][:SMOKE_TASKS]
    initial = sample(args, directory, progress, "initial", tasks, len(tasks), deadline)
    repairs, counts = plan_repairs(tasks, initial)
    threshold = 1 if args.phase == "smoke" else TARGET
    eligible_train = [entry for entry in repairs if entry[1]["split"] == "train"]
    if len(eligible_train) < threshold or (args.phase == "pilot" and counts["sealed"].get("eligible", 0) < threshold):
        return {"verdict": "DATA-NO-GO", "first_attempts": counts, "provenance": inputs}
    if args.phase == "smoke":
        repairs = eligible_train[:GPUS]
    else:
        repairs = eligible_train[:TARGET] + [entry for entry in repairs if entry[1]["split"] == "sealed"]
    size = sum(REPEATS * (len(BRANCHES) if row["eligible"] else 1) for _, row in repairs)
    output = sample(args, directory, progress, "repair", repairs, size, deadline)
    data = join(repairs, output)
    progress.update("running", size, size, detail="pair_analyze")
    if args.phase == "smoke":
        result = {"verdict": "PASS", "completed_branches": size, "complete_crosses": len(data["train"])}
    else:
        result, models, rows = collect_scored(data["train"], data["sealed"])
        atomic_json(directory / "selectors.json", models)
        atomic_json(directory / "sealed_policy_scores.json", rows)
    if monotonic() >= deadline:
        raise TimeoutError("analysis exceeded the frozen wall-clock limit")
    return {**result, "first_attempts": counts, "provenance": inputs, "repair_branches": size,
            "initial_tokens": sum(row["initial_tokens"] for row in initial),
            "panel_tokens": sum(row["tokens"] for row in output)}


def run(args):
    started = monotonic()
    directory = progress_root() / args.id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    progress = Progress(directory, "pair_repair")
    signal.signal(signal.SIGTERM, stop)
    try:
        with progress.heartbeat():
            result = prepare(args, directory, progress) if args.phase == "prepare" else experiment(args, directory, progress, started)
            result.update(phase=args.phase, wall_seconds=monotonic() - started,
                          rank_sha256={path.name: sha256_file(path) for path in sorted(directory.glob("*.jsonl"))})
            atomic_json(directory / SUMMARY, result)
            verdict = result["verdict"].lower().replace("-", "_")
            previous = progress.data["phases"][progress.phase]
            progress.update("done" if verdict in ("ready", "pass", "go") else "blocked",
                            previous["done"], previous["total"], detail=f"pair_{verdict}",
                            metrics={**previous["metrics"], "active_gpus": 0})
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(directory / "incomplete.json", {"verdict": "INCOMPLETE", "phase": args.phase,
                                                    "repo_revision": progress.commit, "error": str(exc)})
        previous = progress.data["phases"].get(progress.phase, {})
        progress.update("failed", previous.get("done", 0), previous.get("total", 0), detail="pair_incomplete")
        raise


def main():
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "pilot", "worker"))
    parser.add_argument("--id")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--smoke-run")
    parser.add_argument("--cpu-jobs", type=int, choices=range(1, 33), default=12, metavar="1..32")
    parser.add_argument("--rank", type=int, choices=range(GPUS), help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("initial", "repair"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "worker":
        rank_worker(args)
        return
    if not RUN_ID.fullmatch(args.id or ""):
        parser.error("--id must be a fresh pair-repair-<name>")
    if args.phase == "prepare" and args.source is None:
        parser.error("prepare requires --source")
    if args.phase in LIMITS and (args.model is None or args.manifest is None):
        parser.error("smoke/pilot require --model and --manifest")
    if args.phase == "pilot" and not RUN_ID.fullmatch(args.smoke_run or ""):
        parser.error("pilot requires --smoke-run pair-repair-<name>")
    run(args)


if __name__ == "__main__":
    main()
