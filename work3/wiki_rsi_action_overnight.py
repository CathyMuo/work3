"""Run two seeded positive / paired comparisons on four independently assigned GPUs."""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

from work3 import wiki_rsi_action_pilot as w6a
from work3 import wiki_rsi_action_scale as scale
from work3 import wiki_rsi_action_train as w6b
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import sha256, title_key
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w6b_overnight_v1"
PREPARED = "w6_overnight_prepared.json"
PREP_SUMMARY = "wiki_rsi_w6_overnight_prepare_summary.json"
SUMMARY = "wiki_rsi_w6_overnight_summary.json"
INTERIM = "wiki_rsi_w6_overnight_interim.json"
SEEDS = (20261119, 20261120)
CHECKPOINTS = (2, 8)
HOURS = 12
JOBS = tuple((seed, arm) for seed in SEEDS for arm in w6b.ARMS[1:])


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    frozen, existing = w6b.prior_inputs(run_dir)
    if not args.retriever or args.retriever_id != frozen["retriever_id"]:
        raise ValueError("Use the original P0 retriever for expanding training")
    w6a_prep = w1.read_json(run_dir / w6a.PREP_SUMMARY)
    checkpoint = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, checkpoint)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    blocked = set(frozen["train_blocked"])
    blocked.update(title_key(name) for case in frozen["evaluation"]
                   for name in [*case["support_titles"], *(title for title, _ in case["visible"])])
    blocked.update(title_key(source) for source in frozen["train_sources"])
    candidate_blocked = blocked | {title_key(row[key]) for row in existing for key in ("source", "target", "foil")}
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    sources, digest, count, by_kind = scale.source_candidates(corpus, candidate_blocked, progress)
    if (digest, count) != (w6a_prep["corpus_sha256"], w6a_prep["corpus_rows"]):
        raise ValueError("Use the byte-identical W6a Wikipedia corpus")
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, frozen["anchors"])
    check = {"protocol": PROTOCOL, "by_dataset": stats, "passed": passed, "retriever_id": args.retriever_id}
    atomic_json(run_dir / "wiki_rsi_w6_overnight_retriever_check.json", check)
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    additions, queries = scale.verify_additions(retriever, sources, existing, blocked, progress,
                                                fits_budget=lambda row: w6b.encode_pair(tok, row) is not None)
    rows = [*({**row, "work_kind": "film"} for row in existing), *additions]
    train, evaluation, overflows = w6b.encode_inputs(tok, rows, frozen["evaluation"], progress)
    atomic_json(run_dir / "w6_overnight_train_pairs.json", {"protocol": PROTOCOL, "records": rows})
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "train": train, "evaluation": evaluation,
                       "retriever_id": frozen["retriever_id"], "anchors": frozen["anchors"]})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "prepared_sha256": sha256(path),
               "model_sha256": checkpoint["model_sha256"], "w6a_prepared_sha256": sha256(run_dir / w6a.PREPARED),
               "w6a_summary_sha256": sha256(run_dir / w6b.W6A_SUMMARY),
               "train_pairs_sha256": sha256(run_dir / "w6_overnight_train_pairs.json"),
               "base_training_pairs": len(existing), "added_training_pairs": len(additions),
               "training_pairs": len(train), "by_work_kind": Counter(row["work_kind"] for row in rows),
               "eligible_sources_by_kind": by_kind, "sources_selected": Counter(kind for kind, _ in sources),
               "physical_queries_excluding_anchors": queries, "evaluation_cases": len(evaluation),
               "context_overflow": overflows, "epochs": list(CHECKPOINTS), "seeds": list(SEEDS), "hours": HOURS}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(evaluation), len(evaluation), detail="ready", metrics={"training_pairs": len(train)})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


class WorkerProgress:
    def __init__(self, path):
        self.path = path

    def update(self, status, done=0, total=0, *, detail="", metrics=None):
        atomic_json(self.path, {"status": status, "done": done, "total": total, "detail": detail,
                                "metrics": metrics or {}})


def worker(config_path):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from work3 import wiki_rsi_action_model as trainer

    config = w1.read_json(config_path)
    folder = config_path.parent
    run_dir, seed, arm = Path(config["run_dir"]), config["seed"], config["arm"]
    if (seed, arm) not in JOBS or torch.cuda.device_count() != 1:
        raise ValueError("Each declared job requires one distinct visible GPU")
    path = run_dir / PREPARED
    if sha256(path) != config["prepared_sha256"]:
        raise ValueError("Worker input changed")
    data = w1.read_json(path)
    progress = WorkerProgress(folder / f"{seed}-{arm}-progress.json")
    progress.update("running", detail="loading_solver")
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(config["model_dir"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config["model_dir"], local_files_only=True,
                                                 torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if model.config.model_type != "qwen2":
        raise ValueError("Expected the frozen P0 Qwen2.5 model")
    adapters = trainer.attach_adapters(model)
    for adapter in adapters:
        adapter.enabled = False
    retriever = w2.Retriever(config["retriever"])
    if (seed, arm) == JOBS[0]:
        rows, queries = w6b.evaluate(model, tok, device, data["evaluation"], retriever, progress, "frozen",
                                      trainer.generate_queries)
        atomic_json(folder / "frozen.json", {"prepared_sha256": config["prepared_sha256"], "rows": rows,
                                             "queries": queries})
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    margins = []
    with torch.no_grad():
        for index, row in enumerate(data["train"], 1):
            values = trainer.mean_logprobs(model, row["prompt"], (row["positive"], row["negative"]), pad, device)
            margins.append(float(values[0] - values[1]))
            progress.update("running", index, len(data["train"]), detail="scoring_action_reference")
    for adapter in adapters:
        adapter.enabled = True

    def checkpoint(epoch, budget):
        rows, queries = w6b.evaluate(model, tok, device, data["evaluation"], retriever, progress, arm,
                                      trainer.generate_queries)
        stem = f"{seed}-{arm}-{epoch}"
        weights = [weight.detach().cpu() for weight in trainer.trainable(adapters)]
        temporary = folder / f"{stem}.pt.tmp"
        torch.save(weights, temporary)
        temporary.replace(folder / f"{stem}.pt")
        atomic_json(folder / f"{stem}.json", {"prepared_sha256": config["prepared_sha256"], "epoch": epoch,
            "seed": seed, "arm": arm, "rows": rows, "queries": queries, "training": budget,
            "torch": torch.__version__, "transformers": transformers.__version__, "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})

    trainer.train_arm(model, adapters, data["train"], margins, pad, device, paired=arm == "paired",
                      epochs=CHECKPOINTS[-1], seed=seed, checkpoints=CHECKPOINTS, on_checkpoint=checkpoint,
                      progress=lambda done, total: progress.update("running", done, total, detail="training_action_pilot"))
    progress.update("done", detail="completed")


def matched_report(folder, evaluation, prepared_sha256, epoch):
    targets = [folder / "frozen.json", *(folder / f"{seed}-{arm}-{epoch}.json" for seed, arm in JOBS)]
    if not all(path.is_file() for path in targets):
        return None
    frozen, *runs = [w1.read_json(path) for path in targets]
    if any(record["prepared_sha256"] != prepared_sha256 or len(record["rows"]) != len(evaluation)
           for record in (frozen, *runs)):
        raise ValueError("Four-arm evaluation outputs do not match the frozen input")
    results = {(run["seed"], run["arm"]): run for run in runs}
    if results.keys() != set(JOBS) or any(run["epoch"] != epoch for run in runs):
        raise ValueError("The four checkpoints are not from the same budget")
    budgets = {tuple(run["training"][key] for key in ("steps", "pair_presentations", "unpadded_input_tokens"))
               for run in runs}
    if len(budgets) != 1:
        raise ValueError("The four checkpoints used different training budgets")
    by_seed = {}
    for seed in SEEDS:
        positive, paired = (results[seed, arm] for arm in w6b.ARMS[1:])
        rows = [{key: case[key] for key in ("dataset", "qid", "title_stratum")} |
                {"arms": {"frozen": frozen["rows"][index], "positive_only": positive["rows"][index],
                          "paired": paired["rows"][index]}} for index, case in enumerate(evaluation)]
        groups, decision = w6b.summarize(rows)
        by_seed[seed] = {"by_dataset_and_title_stratum": groups, "decision": decision,
                         "training": {item["arm"]: item["training"] for item in (positive, paired)},
                         "physical_queries": {item["arm"]: item["queries"] for item in (positive, paired)}}
    passed = all(row["decision"] == "confirm_on_new_natural_pages" for row in by_seed.values())
    return {"epoch": epoch, "frozen_physical_queries": frozen["queries"], "by_seed": by_seed,
            "decision": "confirm_on_new_natural_pages" if passed else "no_go_for_current_paired_training"}


def device_ids(value):
    ids = value.split(",")
    if len(ids) != 4 or any(not re.fullmatch(r"[0-9]+", gpu) for gpu in ids) or len({int(gpu) for gpu in ids}) != 4:
        raise ValueError("Set four distinct physical IDs in WIKI_RSI_OVERNIGHT_GPUS")
    return ids


def terminate(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(args, run_dir, progress):
    devices = device_ids(os.environ.get("WIKI_RSI_OVERNIGHT_GPUS", "0,1,2,3"))
    data, prep = w1.read_json(run_dir / PREPARED), w1.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or prep["protocol"] != PROTOCOL
            or prep["prepared_sha256"] != sha256(run_dir / PREPARED) or data["retriever_id"] != args.retriever_id
            or prep["w6a_prepared_sha256"] != sha256(run_dir / w6a.PREPARED)
            or prep["w6a_summary_sha256"] != sha256(run_dir / w6b.W6A_SUMMARY)
            or prep["train_pairs_sha256"] != sha256(run_dir / "w6_overnight_train_pairs.json")
            or prep["epochs"] != list(CHECKPOINTS) or prep["seeds"] != list(SEEDS) or prep["hours"] != HOURS
            or prep["training_pairs"] != len(data["train"]) or prep["evaluation_cases"] != len(data["evaluation"])):
        raise ValueError("Use the frozen overnight preparation and original W6a evaluation")
    w1.verify_checkpoint(args, prep)
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "passed": passed, "by_dataset": stats, "retriever_id": args.retriever_id}
    atomic_json(run_dir / "wiki_rsi_w6_overnight_run_retriever_check.json", check)
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        return 2
    folder = Path(mkdtemp(prefix="w6_overnight_workers_", dir=run_dir))
    processes, found = [], {}
    timed_out = False
    start = time.monotonic()
    try:
        for (seed, arm), gpu in zip(JOBS, devices, strict=True):
            config = folder / f"{seed}-{arm}-job.json"
            atomic_json(config, {"run_dir": str(run_dir), "seed": seed, "arm": arm,
                                 "model_dir": str(args.model_dir), "prepared_sha256": prep["prepared_sha256"],
                                 "retriever": args.retriever})
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "1"}
            with (folder / f"{seed}-{arm}.log").open("wb") as log:
                processes.append(subprocess.Popen([sys.executable, "-m", __name__, str(config)], env=env,
                                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
        while True:
            for epoch in CHECKPOINTS:
                if epoch not in found:
                    report = matched_report(folder, data["evaluation"], prep["prepared_sha256"], epoch)
                    if report:
                        found[epoch] = report
                        atomic_json(run_dir / INTERIM, {"protocol": PROTOCOL, "worker_directory": folder.name,
                                                       "matched_checkpoints": found})
            codes = [process.poll() for process in processes]
            metrics = {"completed_workers": sum(code == 0 for code in codes), "training_pairs": len(data["train"])}
            progress.update("running", len(found), len(CHECKPOINTS), detail="waiting_overnight_workers", metrics=metrics)
            if all(code is not None for code in codes) or any(code not in (None, 0) for code in codes):
                break
            remaining = HOURS * 3600 - (time.monotonic() - start)
            if remaining <= 0:
                timed_out = True
                break
            time.sleep(min(5, remaining))
    finally:
        terminate(processes)
    # An atomic checkpoint may have been written between polling and shutdown.
    for epoch in CHECKPOINTS:
        if epoch not in found:
            report = matched_report(folder, data["evaluation"], prep["prepared_sha256"], epoch)
            if report:
                found[epoch] = report
    complete = CHECKPOINTS[-1] in found
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "worker_directory": folder.name,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "retriever": check,
               "train_pairs": len(data["train"]), "by_work_kind": prep["by_work_kind"],
               "hours": HOURS, "timed_out": timed_out, "matched_checkpoints": found,
               "jobs": [{"seed": seed, "arm": arm, "gpu": gpu, "exit_code": process.returncode}
                        for (seed, arm), gpu, process in zip(JOBS, devices, processes, strict=True)],
               "wall_seconds": round(time.monotonic() - start, 3),
               "decision": found[CHECKPOINTS[-1]]["decision"] if complete else "incomplete_checkpoints"}
    atomic_json(run_dir / SUMMARY, summary)
    progress.update("done" if complete else "blocked", len(found), len(CHECKPOINTS), detail="completed")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    worker(Path(sys.argv[1]))
