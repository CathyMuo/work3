"""Amplification-gap P0a: does high-budget answer agreement track correctness on labelled math?

Frozen protocol: research/protocols/amp_gap_p0a.md. Factory guide: docs/factory/amp_gap_p0a.md.
Phases: probe-sandbox, verify-data, smoke, p0a, reanalyze (CPU only), worker (internal).
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep, time

from work3.amp_gap_math import PROTOCOL, analyze, code_answer, extract_code, gold_answer, last_boxed, canonical
from work3.pair_sandbox import SandboxStartError, Verifier, sha256_file
from work3.wiki_rsi_progress import REPO, Progress, atomic_json, progress_root

PHASE = "amp_gap"
SUMMARY = "wiki_rsi_amp_gap_summary.json"
RUN_ID = re.compile(r"amp-gap-[a-z0-9][a-z0-9-]{0,48}")
MAX_GPUS = 4
SAMPLES = {"smoke": 2, "p0a": 8}
PER_LEVEL = {"smoke": 3, "p0a": 200}
BUDGETS = {"p0a": {"short": 2048, "force": 2048, "verify": 2048, "code": 1024, "think": 8192},
           "smoke": {"short": 512, "force": 256, "verify": 512, "code": 512, "think": 1024}}
MAX_MODEL_LEN = {False: 8192, True: 12288}
PROMPT_CAP = 1024
CANDIDATE_CAP = 6000
TEMPERATURE, TOP_P = 0.8, 0.95
SEED = 20261127
CHUNK = 64
CODE_LIMITS = {"wall": 6.0, "cpu": 5, "memory": 1 << 30}
CODE_THREADS = 2  # per GPU; namespace creation serializes under load (feedback #62)
LIMITS = {"smoke": 1800, "p0a": 4 * 3600}
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
CODE_SYSTEM = "You are a careful mathematician who solves problems by writing short, correct Python programs."
WAIT = ("\n\nWait, let me carefully re-check every step of the solution above, fix any mistake, "
        "and then state the final answer within \\boxed{}.")
VERIFY = ("{problem}\n\nHere is a candidate solution:\n\n{candidate}\n\nCheck the candidate solution step by step. "
          "If you find an error, correct it and solve the problem again. Put the final answer within \\boxed{{}}.")
CODE = ("Write a Python 3 program that solves the following math problem and prints only the final answer.\n"
        "Use only the Python standard library (math, fractions, itertools, functools, decimal). Prefer exact "
        "arithmetic with integers and fractions.Fraction, and print a fraction as a/b. Put the whole program "
        "in one ```python code block.\n\nProblem: {problem}")
LEVEL = re.compile(r"(\d+)")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stop(signum, _frame):
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def stop_workers(workers):
    for process in workers:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    for process in workers:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)


# ----------------------------------------------------------------------------- data

def data_files(source):
    source = Path(source).resolve(strict=True)
    if source.is_file():
        return source.parent, [source]
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix().lower()
        if path.is_file() and path.suffix in {".parquet", ".jsonl", ".json"} and "train" in relative and "test" not in relative:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"no train .parquet/.jsonl/.json files under {source}")
    return source, files


def read_records(path):
    if path.suffix == ".parquet":
        from pyarrow.parquet import read_table

        return read_table(path).to_pylist()
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", [])


def level_of(record):
    match = LEVEL.search(str(record.get("level") or ""))
    return int(match.group(1)) if match else 0


def load_problems(source):
    base, files = data_files(source)
    problems, hashes, dropped = {}, {}, Counter()
    for path in files:
        hashes[path.relative_to(base).as_posix()] = sha256_file(path)
        for record in read_records(path):
            problem = (record.get("problem") or record.get("question") or "").strip()
            if not problem:
                dropped["no_problem"] += 1
                continue
            gold = gold_answer(record)
            if gold is None:
                dropped["no_gold"] += 1
                continue
            pid = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]
            if pid in problems:
                dropped["duplicate"] += 1
                continue
            problems[pid] = {"id": pid, "problem": problem, "gold": gold, "level": level_of(record),
                             "type": str(record.get("type") or record.get("subject") or "")}
    return list(problems.values()), hashes, dict(dropped)


def select(problems, per_level):
    """Fixed hash order, the same quota per difficulty level (5x quota when no levels exist)."""
    groups = defaultdict(list)
    for row in problems:
        groups[row["level"]].append(row)
    levels = sorted(level for level in groups if level) or [0]
    quota = per_level if levels != [0] else 5 * per_level
    order = lambda row: hashlib.sha256(f"{PROTOCOL}:{row['id']}".encode()).hexdigest()
    return [row for level in levels for row in sorted(groups[level], key=order)[:quota]]


def verify_data(source):
    problems, hashes, dropped = load_problems(source)
    chosen = select(problems, PER_LEVEL["p0a"])
    return {"protocol": PROTOCOL, "files": hashes, "usable_problems": len(problems), "dropped": dropped,
            "levels": dict(sorted(Counter(row["level"] for row in problems).items())),
            "selected_p0a": len(chosen), "selected_levels": dict(sorted(Counter(row["level"] for row in chosen).items())),
            "selection_sha256": hashlib.sha256(json.dumps([row["id"] for row in chosen]).encode()).hexdigest()}


# ----------------------------------------------------------------------------- prompting

class Prompter:
    def __init__(self, model, hybrid):
        from transformers import AutoTokenizer

        self.model = Path(model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True, use_fast=True)
        self.chat = bool(getattr(self.tokenizer, "chat_template", None))
        self.hybrid = hybrid
        if hybrid and not self.chat:
            raise ValueError("--hybrid-thinking needs a tokenizer chat template with enable_thinking")

    def ids(self, user, system=SYSTEM, thinking=False):
        if self.chat:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            extra = {"enable_thinking": thinking} if self.hybrid else {}
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        else:
            text = f"{system}\n\nQuestion: {user}\nAnswer:"
        return self.encode(text)

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def stop_ids(self):
        ids = set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        config = self.model / "generation_config.json"
        if config.is_file():
            value = load(config).get("eos_token_id")
            ids.update(value if isinstance(value, list) else [] if value is None else [value])
        vocab = self.tokenizer.get_vocab()
        ids.update(vocab[token] for token in ("<|im_end|>", "<|endoftext|>") if token in vocab)
        return sorted(ids)


def model_inputs(model):
    model = Path(model).resolve(strict=True)
    weights = sorted(model.glob("*.safetensors"))
    if not weights or not (model / "config.json").is_file():
        raise ValueError(f"expected a local safetensors model with config.json: {model}")
    files = weights + sorted(model.glob("*.json")) + sorted(model.glob("*.jinja")) + sorted(model.glob("*.txt"))
    return model, {path.name: sha256_file(path) for path in files}


# ----------------------------------------------------------------------------- worker

def strip_stop(tokens, stops):
    tokens = list(tokens)
    while tokens and tokens[-1] in stops:
        tokens.pop()
    return tokens


def seed_for(task_id, arm, index=0):
    return int(hashlib.sha256(f"{SEED}:{task_id}:{arm}:{index}".encode()).hexdigest()[:8], 16)


def execute(verifier, code):
    """Run one generated program; retry once on an infrastructure fault before recording `infra`."""
    if code is None:
        return None, "no_code"
    for attempt in range(2):
        try:
            status, stdout = verifier.run(code)
            break
        except (RuntimeError, SandboxStartError):
            if attempt:
                return None, "infra"
            sleep(1.0)
    if status != "ok":
        return None, status
    answer = code_answer(stdout)
    return answer, "ok" if answer is not None else "no_answer"


def worker(args):
    from vllm import LLM, SamplingParams

    plan = load(args.directory / "plan.json")
    settings, rank = plan["settings"], args.rank
    tasks = [row for index, row in enumerate(plan["problems"]) if index % settings["gpus"] == rank]
    budgets, k, hybrid = settings["budgets"], settings["samples"], settings["hybrid"]
    prompter = Prompter(plan["model"], hybrid)
    stops = prompter.stop_ids()
    wait_ids = prompter.encode(WAIT)
    engine = LLM(model=plan["model"], tokenizer=plan["model"], dtype="bfloat16", max_model_len=settings["max_model_len"],
                 gpu_memory_utilization=settings["gpu_memory"], seed=SEED + rank, enable_prefix_caching=True,
                 generation_config="vllm")
    verifier = Verifier(**CODE_LIMITS)

    def params(arm, n, seed):
        return SamplingParams(n=n, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=budgets[arm], seed=seed,
                              stop_token_ids=stops, skip_special_tokens=True)

    def generate(prompts, sampling):
        outputs = engine.generate([{"prompt_token_ids": ids} for ids in prompts], sampling_params=sampling, use_tqdm=False)
        return [[(choice.text, list(choice.token_ids), choice.finish_reason) for choice in output.outputs]
                for output in outputs]

    done, generated = 0, 0
    status_file = args.directory / f"status-{rank}.json"
    with (args.directory / f"rank-{rank}.jsonl").open("x", encoding="utf-8") as stream, \
            gzip.open(args.directory / f"texts-{rank}.jsonl.gz", "wt", encoding="utf-8") as texts, \
            ThreadPoolExecutor(max_workers=CODE_THREADS) as pool:
        for offset in range(0, len(tasks), CHUNK):
            chunk = tasks[offset:offset + CHUNK]
            prompts = [prompter.ids(task["problem"]) for task in chunk]
            short = generate(prompts, [params("short", k, seed_for(task["id"], "short")) for task in chunk])
            force_prompts, verify_prompts, force_params, verify_params = [], [], [], []
            for task, ids, outputs in zip(chunk, prompts, short, strict=True):
                for index, (text, tokens, _) in enumerate(outputs):
                    force_prompts.append(ids + strip_stop(tokens, stops) + wait_ids)
                    force_params.append(params("force", 1, seed_for(task["id"], "force", index)))
                    verify_prompts.append(prompter.ids(VERIFY.format(problem=task["problem"], candidate=text[-CANDIDATE_CAP:])))
                    verify_params.append(params("verify", 1, seed_for(task["id"], "verify", index)))
            force = generate(force_prompts, force_params)
            verify = generate(verify_prompts, verify_params)
            code_prompts = [prompter.ids(CODE.format(problem=task["problem"]), system=CODE_SYSTEM) for task in chunk]
            code = generate(code_prompts, [params("code", k, seed_for(task["id"], "code")) for task in chunk])
            programs = [extract_code(text) for outputs in code for text, _, _ in outputs]
            executed = list(pool.map(lambda program: execute(verifier, program), programs))
            think = None
            if hybrid:
                think_prompts = [prompter.ids(task["problem"], thinking=True) for task in chunk]
                think = generate(think_prompts, [params("think", k, seed_for(task["id"], "think")) for task in chunk])
            for position, (task, ids) in enumerate(zip(chunk, prompts, strict=True)):
                window = slice(position * k, (position + 1) * k)
                short_out = short[position]
                force_out = [choice[0] for choice in force[window]]
                verify_out = [choice[0] for choice in verify[window]]
                code_out, runs = code[position], executed[window]
                arms = {
                    "short": [(canonical(last_boxed(text)), len(tokens), finish) for text, tokens, finish in short_out],
                    "force": [(canonical(last_boxed(base[0] + WAIT + text)), len(tokens), finish)
                              for base, (text, tokens, finish) in zip(short_out, force_out, strict=True)],
                    "verify": [(canonical(last_boxed(text)), len(tokens), finish) for text, tokens, finish in verify_out],
                    "code": [(answer, len(tokens), outcome if finish != "length" else f"{outcome}+length")
                             for (_, tokens, finish), (answer, outcome) in zip(code_out, runs, strict=True)],
                }
                if hybrid:
                    arms["think"] = [(canonical(last_boxed(text)), len(tokens), finish)
                                     for text, tokens, finish in think[position]]
                row = {"id": task["id"], "level": task["level"], "type": task["type"], "gold": task["gold"],
                       "prompt_tokens": len(ids),
                       "arms": {arm: {"answers": [item[0] for item in items], "tokens": [item[1] for item in items],
                                      "finish": [item[2] for item in items]} for arm, items in arms.items()}}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                texts.write(json.dumps({"id": task["id"], "short": [c[0] for c in short_out], "force": [c[0] for c in force_out],
                                        "verify": [c[0] for c in verify_out], "code": [c[0] for c in code_out],
                                        "think": [c[0] for c in think[position]] if hybrid else None},
                                       ensure_ascii=False) + "\n")
                generated += sum(sum(arm["tokens"]) for arm in row["arms"].values())
                done += 1
            stream.flush()
            atomic_json(status_file, {"done": done, "total": len(tasks), "generated_tokens": generated,
                                      "sandbox": verifier.timing(), "updated_at": int(time())})


# ----------------------------------------------------------------------------- orchestration

def visible_gpus(requested):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [value.strip() for value in devices.split(",") if value.strip()] if devices else [str(i) for i in range(requested)]
    if len(ids) < requested:
        raise RuntimeError(f"--gpus {requested} but only {len(ids)} visible devices: {devices!r}")
    return ids[:requested]


def provenance(args, problems, hashes, verifier, prompter):
    model, model_hashes = model_inputs(args.model)
    versions = {}
    for package in ("vllm", "torch", "transformers", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    if versions["vllm"] is None:
        raise RuntimeError("vLLM is required; see the factory guide for the environment")
    return {"protocol": PROTOCOL, "model": str(model), "model_sha256": model_hashes, "hybrid_thinking": args.hybrid_thinking,
            "chat_template": prompter.chat, "data_files": hashes, "versions": versions,
            "python": sys.version.split()[0], "sandbox": verifier.version(),
            "selection_sha256": hashlib.sha256(json.dumps([row["id"] for row in problems]).encode()).hexdigest()}


def collect(directory, progress, workers, total, deadline, gpus):
    while True:
        codes = [process.poll() for process in workers]
        states = [load(path) for path in sorted(directory.glob("status-*.json"))]
        done = sum(state["done"] for state in states)
        tokens = sum(state["generated_tokens"] for state in states)
        rows = [{"name": f"rank-{index}", "device": gpus[index], "detail": "amp_generating", "done": state["done"],
                 "total": state["total"], "metrics": {"generated_tokens": state["generated_tokens"]},
                 "updated_at": state["updated_at"]} for index, state in enumerate(states)]
        progress.update("running", done, total, detail="amp_generating", workers=rows,
                        metrics={"active_gpus": len(gpus), "generated_tokens": tokens})
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError(f"worker exited with {codes}; inspect worker-<rank>.log")
        if all(code == 0 for code in codes):
            return
        if monotonic() >= deadline:
            raise TimeoutError("wall-clock limit reached")
        sleep(5)


def experiment(args, directory, progress, started):
    deadline = started + LIMITS[args.phase]
    progress.update("running", detail="amp_preflight")
    verifier = Verifier(**CODE_LIMITS)
    probe = verifier.probe()
    problems, hashes, dropped = load_problems(args.data)
    prompter = Prompter(args.model, args.hybrid_thinking)
    chosen = select(problems, PER_LEVEL[args.phase])
    lengths = {row["id"]: len(prompter.ids(row["problem"])) for row in chosen}
    eligible = [row for row in chosen if lengths[row["id"]] <= PROMPT_CAP]
    inputs = provenance(args, eligible, hashes, verifier, prompter)
    if args.phase == "p0a":
        smoke = load(progress_root() / args.smoke_run / SUMMARY)
        same = {key: inputs[key] for key in ("model_sha256", "hybrid_thinking", "data_files", "versions", "sandbox")}
        if smoke.get("phase") != "smoke" or smoke.get("verdict") != "PASS" or \
                {key: smoke["provenance"].get(key) for key in same} != same or \
                smoke["provenance"].get("repo_revision") != progress.commit:
            raise ValueError("p0a requires a PASS smoke from this commit with the same model, data, runtime and sandbox")
    inputs["repo_revision"] = progress.commit
    gpus = visible_gpus(args.gpus)
    settings = {"gpus": len(gpus), "samples": SAMPLES[args.phase], "budgets": BUDGETS[args.phase],
                "hybrid": args.hybrid_thinking, "max_model_len": MAX_MODEL_LEN[args.hybrid_thinking],
                "gpu_memory": args.gpu_memory, "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED,
                "code_limits": CODE_LIMITS, "prompt_cap": PROMPT_CAP}
    atomic_json(directory / "inputs.json", {"provenance": inputs, "settings": settings, "probe": probe,
                                           "dropped": dropped, "prompt_cap_exclusions": len(chosen) - len(eligible)})
    atomic_json(directory / "plan.json", {"model": inputs["model"], "settings": settings, "problems": eligible})
    workers = []
    try:
        for rank, device in enumerate(gpus):
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
            command = [sys.executable, "-m", "work3.amp_gap", "worker", "--rank", str(rank), "--directory", str(directory)]
            with (directory / f"worker-{rank}.log").open("wb") as log:
                workers.append(subprocess.Popen(command, cwd=REPO, env=environment, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True))
        collect(directory, progress, workers, len(eligible), deadline, gpus)
    finally:
        stop_workers(workers)
    rows = read_rows(directory, len(gpus))
    if sorted(row["id"] for row in rows) != sorted(row["id"] for row in eligible):
        raise ValueError(f"collected {len(rows)} records for {len(eligible)} planned problems")
    progress.update("running", len(rows), len(rows), detail="amp_analyze")
    result = analyze(rows)
    sandbox = [load(path)["sandbox"] for path in sorted(directory.glob("status-*.json"))]
    result.update(provenance=inputs, settings=settings, sandbox_timing=sandbox,
                  prompt_cap_exclusions=len(chosen) - len(eligible))
    if args.phase == "smoke":
        code_ok = sum(finish.startswith("ok") for row in rows for finish in row["arms"]["code"]["finish"])
        answered = all(any(answer is not None for row in rows for answer in row["arms"][arm]["answers"])
                       for arm in rows[0]["arms"])
        infra = max(arm["infra_rate"] or 0 for arm in result["arms"].values())
        result["p0a_gate_preview"] = result.pop("verdict")
        result["verdict"] = "PASS" if answered and code_ok > 0 and infra <= 0.01 else "FAIL"
        result["smoke_checks"] = {"every_arm_answered": answered, "executed_programs_ok": code_ok, "max_infra_rate": infra}
    return result


def read_rows(directory, gpus):
    rows = []
    for rank in range(gpus):
        with (directory / f"rank-{rank}.jsonl").open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def run(args):
    if os.name != "posix":
        raise RuntimeError("run on the factory Linux environment")
    started = monotonic()
    directory = progress_root() / args.id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    progress = Progress(directory, PHASE)
    signal.signal(signal.SIGTERM, stop)
    try:
        with progress.heartbeat():
            result = experiment(args, directory, progress, started)
            result.update(phase=args.phase, wall_seconds=round(monotonic() - started, 1),
                          rank_sha256={path.name: sha256_file(path) for path in sorted(directory.glob("rank-*.jsonl"))})
            atomic_json(directory / SUMMARY, result)
            verdict = result["verdict"].lower().replace("-", "_")
            previous = progress.data["phases"][PHASE]
            progress.update("done" if verdict in ("pass", "go") else "blocked", previous["done"], previous["total"],
                            detail=f"amp_{verdict}", metrics={**previous["metrics"], "active_gpus": 0})
            print(json.dumps({key: result[key] for key in ("phase", "verdict", "passing_cells", "problems", "wall_seconds")
                              if key in result}, ensure_ascii=False, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(directory / "incomplete.json", {"verdict": "INCOMPLETE", "phase": args.phase,
                                                    "repo_revision": progress.commit, "error": str(exc)})
        previous = progress.data["phases"].get(PHASE, {})
        progress.update("failed", previous.get("done", 0), previous.get("total", 0), detail="amp_incomplete")
        raise


def reanalyze(run_id):
    directory = progress_root() / run_id
    settings = load(directory / "inputs.json")["settings"]
    result = analyze(read_rows(directory, settings["gpus"]))
    atomic_json(directory / "reanalysis.json", result)
    return result


def main():
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe-sandbox", "verify-data", "smoke", "p0a", "reanalyze", "worker"))
    parser.add_argument("--id")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoke-run")
    parser.add_argument("--hybrid-thinking", action="store_true",
                        help="model has a thinking switch (e.g. Qwen3-4B); adds the thinking-mode arm")
    parser.add_argument("--gpus", type=int, choices=range(1, MAX_GPUS + 1), default=MAX_GPUS)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "worker":
        worker(args)
        return
    if args.phase == "probe-sandbox":
        verifier = Verifier(**CODE_LIMITS)
        print(json.dumps({"probe": verifier.probe(), "version": verifier.version()}, ensure_ascii=False, indent=2))
        return
    if args.phase == "verify-data":
        if args.data is None:
            parser.error("verify-data requires --data")
        print(json.dumps(verify_data(args.data), ensure_ascii=False, indent=2))
        return
    if not RUN_ID.fullmatch(args.id or ""):
        parser.error("--id must be amp-gap-<name> (lowercase letters, digits, dashes)")
    if args.phase == "reanalyze":
        result = reanalyze(args.id)
        print(json.dumps({key: result[key] for key in ("verdict", "passing_cells", "problems")}, indent=2))
        return
    if args.data is None or args.model is None:
        parser.error(f"{args.phase} requires --data and --model")
    if args.phase == "p0a" and not RUN_ID.fullmatch(args.smoke_run or ""):
        parser.error("p0a requires --smoke-run amp-gap-<name> of a PASS smoke")
    if not 0.3 <= args.gpu_memory <= 0.95:
        parser.error("--gpu-memory must be within 0.3..0.95")
    run(args)


if __name__ == "__main__":
    main()
