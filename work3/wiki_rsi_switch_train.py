"""Run a frozen-reader gate and a four-GPU comparison of query preference objectives."""

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

from work3 import wiki_rsi_action_overnight as parallel
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_switch_data as data
from work3 import wiki_rsi_two_query_live as reader
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import sha256
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import atomic_json

EPOCHS = 6
ARMS = ("ordinary", "interaction")
JOBS = tuple((seed, arm) for seed in parallel.SEEDS for arm in ARMS)
SUMMARY = "wiki_rsi_switch_summary.json"
GATE = "wiki_rsi_switch_reader_gate_summary.json"
ENCODED = "switch_encoded.json"


def load_solver(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise ValueError("Each worker needs one visible H800")
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(device)
    return tok, model, device


def reader_matrix(row, tok, model, device, generate):
    prompts = [reader.pack(tok, row["question"], [*visible, *returned], reader.answer_prompt, reader.ANSWER_TOKENS)[0]
               for visible in row["states"] for returned in row["returns"]]
    generated = generate(model, tok, prompts, device, reader.ANSWER_TOKENS)
    if len(generated) != 4:
        raise ValueError("Expected all four frozen reader outcomes")
    scores = [replay.score_generation(item["text"], row["aliases"], item["finish"]) for item in generated]
    outcomes = [row["correct"] for row in scores]
    return [outcomes[:2], outcomes[2:]], all(row["valid"] for row in scores)


def gated(matrix):
    return matrix == [[0, 1], [1, 0]]


def gate(config, folder):
    from work3 import wiki_rsi_action_model as model_tools
    from work3 import wiki_rsi_evidence_switch as switch

    source = Path(config["input"])
    if sha256(source) != config["digest"]:
        raise ValueError("The source pairs changed")
    records = replay.read_json(source)["records"]
    tok, model, device = load_solver(config["model_dir"])
    kept, encoded, wins, complete = {}, {}, {}, {}
    for split, rows in records.items():
        encoded[split], kept[split], wins[split], complete[split] = [], Counter(), Counter(), Counter()
        for index, row in enumerate(rows, 1):
            inp = switch.encode_example(tok, row)
            if inp is None:
                continue
            matrix, valid = reader_matrix(row, tok, model, device, model_tools.generate_queries)
            usable = valid and gated(matrix)
            wins[split][row["relation"]] += usable
            complete[split][row["relation"]] += valid
            # Keep all encodable test cases, including ones where the reader fails with enough evidence.
            if split == "test" or usable:
                encoded[split].append({**inp, "reader": matrix, "relation": row["relation"]})
                kept[split][row["relation"]] += 1
            if index % 16 == 0:
                print(f"{split} {index}/{len(rows)} checked; {len(encoded[split])} retained", flush=True)
    path = folder / ENCODED
    atomic_json(path, {"protocol": data.PROTOCOL, "records": encoded})
    receipt = {"protocol": data.PROTOCOL, "encoded_sha256": sha256(path),
               "source_pairs": {split: len(rows) for split, rows in records.items()},
               "retained": kept, "reader_flip": wins, "all_four_answers_valid": complete,
               "ready": len(encoded["train"]) >= data.MIN_TRAIN and len(encoded["test"]) >= data.MIN_TEST
                        and len(kept["train"]) >= 2}
    atomic_json(folder / GATE, receipt)


def train(model, adapters, rows, device, pad, seed, arm):
    import torch

    from work3 import wiki_rsi_action_model as model_tools
    from work3 import wiki_rsi_evidence_switch as switch

    model.train()
    params = model_tools.trainable(adapters)
    optimizer = torch.optim.AdamW(params, lr=model_tools.LEARNING_RATE, weight_decay=0)
    schedule = model_tools.scheduled_indices(len(rows), epochs=EPOCHS, seed=seed)
    tokens, count = 0, 0
    for step, group in enumerate(schedule, 1):
        for index in group:
            case = rows[index]
            scores = switch.score_example(model, case, pad, device)
            ordinary, interaction = switch.losses(scores)
            loss = interaction if arm == "interaction" else ordinary + interaction * 0
            (loss / len(group)).backward()
            tokens += 2 * sum(map(len, case["prompts"])) + 2 * sum(map(len, case["completions"]))
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        count += len(group)
        if step % 50 == 0:
            print(f"step {step}/{len(schedule)}", flush=True)
    return {"steps": len(schedule), "presentations": count, "tokens": tokens, "epochs": EPOCHS}


def measure_choices(choices, reader_outcomes):
    return {"both_correct": int(choices == [1, 0]),
            "covered": int(choices[0] == 1) + int(choices[1] == 0),
            "em": sum(reader_outcomes[state][choice] for state, choice in enumerate(choices))}


def evaluate(model, rows, pad, device):
    import torch

    from work3 import wiki_rsi_evidence_switch as switch

    model.eval()
    report = {rel: Counter() for rel in ("all", *data.RELATIONS)}
    with torch.no_grad():
        for row in rows:
            choices = switch.score_example(model, row, pad, device).argmax(dim=-1).tolist()
            result = measure_choices(choices, row["reader"])
            result["pairs"] = 1
            report["all"].update(result)
            report[row["relation"]].update(result)
    return report


def fit(config, folder):
    import torch

    from work3 import wiki_rsi_action_model as model_tools

    path = folder / ENCODED
    if sha256(path) != config["encoded_sha256"]:
        raise ValueError("Worker output from the reader gate changed")
    groups = replay.read_json(path)["records"]
    seed, arm = config["seed"], config["arm"]
    if (seed, arm) not in JOBS:
        raise ValueError("Unexpected training condition")
    torch.manual_seed(seed)
    tok, model, device = load_solver(config["model_dir"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    adapters = model_tools.attach_adapters(model)
    if (seed, arm) == JOBS[0]:
        for adapter in adapters:
            adapter.enabled = False
        atomic_json(folder / "frozen.json", evaluate(model, groups["test"], pad, device))
        for adapter in adapters:
            adapter.enabled = True
    budget = train(model, adapters, groups["train"], device, pad, seed, arm)
    results = evaluate(model, groups["test"], pad, device)
    stem = f"{seed}-{arm}"
    torch.save([param.detach().cpu() for param in model_tools.trainable(adapters)], folder / f"{stem}.pt")
    atomic_json(folder / f"{stem}.json", {"encoded_sha256": config["encoded_sha256"],
                "training": budget, "held_out": results})


def launch(folder, name, config, gpu):
    path = folder / f"{name}-job.json"
    atomic_json(path, config)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1"}
    with (folder / f"{name}.log").open("wb") as log:
        return subprocess.Popen([sys.executable, "-m", __name__, name, str(path)], env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def run(args, run_dir, progress):
    devices = parallel.device_ids(os.environ.get("WIKI_RSI_SWITCH_GPUS", "0,1,2,3"))
    path = run_dir / data.PREPARED
    ready, prepared = [replay.read_json(run_dir / name) for name in (data.PREP_SUMMARY, data.PREPARED)]
    if (ready["protocol"] != data.PROTOCOL or prepared["protocol"] != data.PROTOCOL
            or ready["prepared_sha256"] != sha256(path)
            or ready["w7b_sha256"] != sha256(run_dir / reader.SUMMARY)
            or prepared["retriever_id"] != args.retriever_id):
        raise ValueError("Use the pinned W7b and evidence-switch artifacts")
    replay.verify_checkpoint(args, ready)
    if not ready["ready_for_reader"]:
        report = {"protocol": data.PROTOCOL, "repo_revision": revision(wiki.ROOT),
                  "preparation_sha256": sha256(run_dir / data.PREP_SUMMARY),
                  "pairs": ready["pairs"], "decision": "insufficient_retrieval_verified_pairs"}
        atomic_json(run_dir / SUMMARY, report)
        progress.update("blocked", detail=report["decision"])
        print(json.dumps(report, indent=2), flush=True)
        return 2
    folder = Path(mkdtemp(prefix="switch_workers_", dir=run_dir))
    base = {"model_dir": str(args.model_dir), "input": str(path), "digest": ready["prepared_sha256"]}
    start = time.monotonic()
    deadline = start + parallel.HOURS * 3600
    progress.update("running", detail="gating_on_frozen_reader")
    gate_job = launch(folder, "gate", base, devices[0])
    try:
        gate_job.wait(timeout=max(0, deadline - time.monotonic()))
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        parallel.terminate([gate_job])
        raise
    if gate_job.returncode != 0:
        raise RuntimeError(f"Frozen reader gate failed; see {folder.name}/gate.log")
    receipt = replay.read_json(folder / GATE)
    atomic_json(run_dir / GATE, receipt)
    report = {"protocol": data.PROTOCOL, "repo_revision": revision(wiki.ROOT),
              "preparation_sha256": sha256(run_dir / data.PREP_SUMMARY),
              "worker_directory": folder.name, "reader_gate": receipt, "new_physical_queries": 0}
    if not receipt["ready"]:
        report["decision"] = "insufficient_reader_verified_pairs"
        atomic_json(run_dir / SUMMARY, report)
        progress.update("blocked", detail="insufficient_reader_verified_pairs")
        print(json.dumps(report, indent=2), flush=True)
        return 2
    processes = []
    try:
        for (seed, arm), gpu in zip(JOBS, devices, strict=True):
            config = {**base, "seed": seed, "arm": arm, "encoded_sha256": receipt["encoded_sha256"]}
            processes.append(launch(folder, f"{seed}-{arm}", config, gpu))
        while True:
            codes = [job.poll() for job in processes]
            progress.update("running", codes.count(0), len(JOBS), detail="training_evidence_switch")
            remaining = deadline - time.monotonic()
            if any(code not in (0, None) for code in codes) or all(code == 0 for code in codes) or remaining <= 0:
                break
            time.sleep(min(3, remaining))
    finally:
        parallel.terminate(processes)
    results = {}
    completed = all(job.returncode == 0 for job in processes)
    if completed:
        for seed, arm in JOBS:
            name = f"{seed}-{arm}"
            results[name] = replay.read_json(folder / f"{name}.json")
        budgets = [row["training"] for row in results.values()]
        if (any(row["encoded_sha256"] != receipt["encoded_sha256"] for row in results.values())
                or any(budget != budgets[0] for budget in budgets[1:])):
            raise ValueError("The four jobs used different examples or training budgets")
    report.update({"decision": "held_out_pair_results_ready" if completed else "incomplete_workers",
                   "jobs": [{"seed": seed, "arm": arm, "gpu": gpu, "exit_code": job.returncode}
                            for (seed, arm), gpu, job in zip(JOBS, devices, processes, strict=True)],
                   "results": results, "wall_seconds": round(time.monotonic() - start, 3)})
    if completed:
        report["frozen_test"] = replay.read_json(folder / "frozen.json")
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done" if completed else "blocked", sum(job.returncode == 0 for job in processes), len(JOBS),
                    detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if completed else 2


if __name__ == "__main__":
    job = Path(sys.argv[2])
    task = gate if sys.argv[1] == "gate" else fit
    task(replay.read_json(job), job.parent)
