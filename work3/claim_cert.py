"""Plan B, first experiment (retrieve-to-certify): at the same certified error rate and the same retrieval cost, does
spending extra retrieval on uncertain claims keep more true claims than reading more passages for every claim?

Claims are Qwen3-4B's closed-book PopQA answers from answer-prior-run-v1, true or false by the PopQA aliases; passages
come from the stored wiki-18 E5 vectors. Frozen protocol: research/protocols/claim_cert_b0.md. Factory guide:
docs/factory/claim_cert_b0.md. Phases: smoke and run (4 GPUs), worker (internal).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from pathlib import Path
from time import monotonic

from work3.answer_prior import (PREPARE_ID, PREPARE_SUMMARY, Stages, model_inputs, stop_workers, versions,
                                visible_gpus, wait_all)
from work3.answer_prior_text import PROTOCOL as SOURCE_PROTOCOL
from work3.answer_prior_text import digest
from work3.claim_cert_stats import (ALPHAS, CONFIGS, JUDGE_CUT, PROTOCOL, SPLITS, aggregate, auc, decide,
                                    judge_calibration, judge_errors, rounded, run_split)
from work3.wiki_rsi_progress import REPO, Progress, atomic_json, progress_root

PHASE = "claim_cert"
SUMMARY = "wiki_rsi_claim_cert_summary.json"
SOURCE_SUMMARY = "wiki_rsi_answer_prior_summary.json"
RUN_ID = re.compile(r"claim-cert-[a-z0-9][a-z0-9-]{0,48}")
MAX_GPUS = 4
SMOKE_CLAIMS = 400
LIMITS = {"smoke": 90 * 60, "run": 6 * 3600}
MIN = {"run": {"claims": 10000, "true": 1000, "parsed": 0.95}, "smoke": {"claims": 200, "true": 10, "parsed": 0.90}}
AUC_FLOOR = 0.60
STAGE_PLAN = {
    "run": [("preflight", 120, False, False), ("search", 300, False, False), ("verify", 1500, False, False),
            ("judge", 1800, False, False), ("analysis", 600, False, False)],
    "smoke": [("preflight", 120, False, False), ("search", 240, False, False), ("verify", 300, False, False),
              ("judge", 300, False, False), ("analysis", 60, False, False)],
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stop(signum, _frame):
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def spawn(kind, plan, rank, devices, directory):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": devices, "TOKENIZERS_PARALLELISM": "true"}
    command = [sys.executable, "-m", "work3.claim_cert", "worker", "--kind", kind, "--plan", str(plan),
               "--rank", str(rank)]
    log = (directory / f"worker-{kind}-{rank}.log").open("wb")
    try:
        return subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    finally:
        log.close()


def worker_main(args):
    from work3 import claim_cert_workers as workers

    if args.kind == "search":
        workers.search_worker(args)
    elif args.kind in ("verify", "judge"):
        workers.verify_worker(args)
    else:
        raise ValueError(args.kind)


# ----------------------------------------------------------------------------- inputs

def build_claims(questions, knownness, phase):
    """One claim per question: Qwen3-4B's cleaned greedy answer, true by the PopQA alias rule; abstentions dropped."""
    claims = []
    for row, question in enumerate(questions):
        info = knownness.get(question["qid"])
        if info is None or info["abstains"] or not info["answer"].strip():
            continue
        claims.append({"qid": question["qid"], "row": row, "question": question["question"], "answer": info["answer"],
                       "truth": bool(info["greedy_correct"]), "stratum": info["stratum"], "prop": question["prop"]})
    if phase == "smoke":
        claims = sorted(claims, key=lambda claim: digest("claim_cert_smoke", claim["qid"]))[:SMOKE_CLAIMS]
        claims.sort(key=lambda claim: claim["row"])
    return claims


def stored_shards(index_root):
    """Manifests of the E5 shards written by answer-prior-run-v1, in shard order."""
    directory = Path(index_root) / "e5"
    manifests = sorted(directory.glob("shard-*.json"), key=lambda path: int(path.stem.split("-")[1]))
    if not manifests:
        raise FileNotFoundError(f"no stored E5 shards under {directory}")
    shards = []
    for path in manifests:
        manifest = load(path)
        vectors = path.with_suffix(".npy")
        if not manifest.get("complete") or not vectors.is_file():
            raise ValueError(f"{path} is incomplete or its vectors are missing")
        shards.append({"manifest": str(path), "vectors": str(vectors), "data": manifest})
    return shards


# ----------------------------------------------------------------------------- stages

def search_stage(context, claims, shards):
    import numpy as np

    directory, gpus, stages = context["directory"], context["gpus"], context["stages"]
    queries = [claim["question"] for claim in claims] + [f"{claim['question']} {claim['answer']}" for claim in claims]
    atomic_json(directory / "queries.json", queries)
    expected = {key: shards[0]["data"][key] for key in ("protocol", "model_sha256", "corpus_range_sha256",
                                                        "encoder_version")}
    if expected["model_sha256"] != context["model_sha256"]["e5"]:
        raise ValueError("--e5 differs from the model that encoded the stored shards")
    if expected["corpus_range_sha256"] != context["corpus_sha256"]:
        raise ValueError("the stored shards were encoded from a different corpus")
    assignments = [[{"manifest": shard["manifest"], "vectors": shard["vectors"]} for shard in shards[rank::len(gpus)]]
                   for rank in range(len(gpus))]
    plan = directory / "plan-search.json"
    atomic_json(plan, {"model": context["models"]["e5"], "queries": str(directory / "queries.json"),
                       "assignments": assignments, "expected_manifest": expected, "directory": str(directory)})
    stages.start("search", total=len(shards))
    workers = []
    try:
        workers = [spawn("search", plan, rank, device, directory) for rank, device in enumerate(gpus)
                   if assignments[rank]]
        wait_all(workers, context, "claim_cert_search", "status-search-*.json", "search")
    finally:
        stop_workers(workers)
    parts = [np.load(directory / f"search-part-{rank}.npz") for rank in range(len(gpus)) if assignments[rank]]
    ids = np.concatenate([part["ids"] for part in parts], axis=1)
    scores = np.concatenate([part["scores"] for part in parts], axis=1)
    scores = np.where(ids >= 0, scores, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :20]
    ids = np.take_along_axis(ids, order, 1)
    count = len(claims)
    # rows follow the claims list (verify workers index them by position, not by the question row)
    q_ids, qa_ids = ids[:count], ids[count:]
    np.savez(directory / "search.npz", q_ids=q_ids, qa_ids=qa_ids)
    stages.finish("search", note=f"{len(queries):,} 条查询 · {len(shards)} 个已存分片")
    return q_ids, qa_ids


def verify_stage(context, claims, tag, model, tp, configs):
    directory, gpus, stages = context["directory"], context["gpus"], context["stages"]
    groups = [",".join(gpus[index:index + tp]) for index in range(0, len(gpus) - tp + 1, tp)]
    payload = [{"qid": claim["qid"], "row": position, "question": claim["question"], "answer": claim["answer"]}
               for position, claim in enumerate(claims)]
    atomic_json(directory / f"claims-{tag}.json", payload)
    plan = directory / f"plan-{tag}.json"
    atomic_json(plan, {"model": model, "tag": tag, "tp": tp, "ranks": len(groups), "configs": list(configs),
                       "claims": str(directory / f"claims-{tag}.json"), "search": str(directory / "search.npz"),
                       "corpus": context["corpus"], "offsets": context["offsets"], "directory": str(directory),
                       "gpu_memory": context["gpu_memory"], "max_num_seqs": context["max_num_seqs"],
                       "max_batched_tokens": context["max_batched_tokens"], "chunk": context["chunk"]})
    stages.start(tag, total=len(claims))
    workers = []
    try:
        workers = [spawn(tag, plan, rank, devices, directory) for rank, devices in enumerate(groups)]
        wait_all(workers, context, f"claim_cert_{tag}", f"status-{tag}-*.json", tag)
    finally:
        stop_workers(workers)
    rows = {}
    for rank in range(len(groups)):
        with (directory / f"{tag}-{rank}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                rows[item["qid"]] = item
    if set(rows) != {claim["qid"] for claim in claims}:
        raise ValueError(f"{tag}: {len(rows)} scored claims for {len(claims)} planned")
    truncated = sum(load(path).get("truncated", 0) for path in directory.glob(f"status-{tag}-*.json"))
    parsed = {key: sum(rows[claim["qid"]]["parsed"][key] for claim in claims) / len(claims) for key in configs}
    stages.finish(tag, note="解析率 " + " / ".join(f"{key} {value:.3f}" for key, value in parsed.items()))
    return rows, {"parsed": {key: rounded(value) for key, value in parsed.items()}, "truncated_prompts": truncated,
                  "groups": groups}


_CLAIMS = None


def _init(claims):
    global _CLAIMS
    _CLAIMS = claims


def _split(job):
    repeat, label_key = job
    return run_split(_CLAIMS, repeat, label_key)


def analysis(context, claims, verified, judged, phase):
    records = []
    for claim in claims:
        row = {"qid": claim["qid"], "truth": claim["truth"], "stratum": claim["stratum"], "prop": claim["prop"],
               "scores": verified[claim["qid"]]["scores"]}
        row["self_judge"] = row["scores"]["q20"] >= JUDGE_CUT
        if judged is not None:
            row["judge_score"] = judged[claim["qid"]]["scores"]["q20"]
            row["judge"] = row["judge_score"] >= JUDGE_CUT
        records.append(row)
    repeats = range(SPLITS[phase])
    jobs = [(repeat, "truth") for repeat in repeats] + [(repeat, "self_judge") for repeat in repeats]
    if judged is not None:
        jobs += [(repeat, "judge") for repeat in repeats]
    with ProcessPoolExecutor(max_workers=max(1, min(32, os.cpu_count() or 1)), initializer=_init,
                             initargs=(records,)) as pool:
        results = list(pool.map(_split, jobs))
    by_label = {}
    for (repeat, label), result in zip(jobs, results):
        by_label.setdefault(label, []).append(result)
    truth = [row["truth"] for row in records]
    descriptive = {key: {"auc": rounded(auc([row["scores"][key] for row in records], truth)),
                         "mean_true": rounded(sum(row["scores"][key] for row in records if row["truth"]) /
                                              max(1, sum(truth))),
                         "mean_false": rounded(sum(row["scores"][key] for row in records if not row["truth"]) /
                                               max(1, len(truth) - sum(truth))),
                         "fp_at_half": {stratum: rounded(
                             sum(row["scores"][key] >= JUDGE_CUT for row in records
                                 if not row["truth"] and row["stratum"] == stratum) /
                             max(1, sum(1 for row in records if not row["truth"] and row["stratum"] == stratum)))
                             for stratum in ("wrong", "unknown", "partial")}}
                   for key in CONFIGS}
    g1_rows = aggregate(by_label["truth"])
    self_judge = {"errors": judge_errors(records, "self_judge"),
                  "calibration": judge_calibration(by_label["self_judge"])}
    external = None
    if judged is not None:
        external = {"errors": judge_errors(records, "judge"), "calibration": judge_calibration(by_label["judge"])}
    return records, {"g1": g1_rows, "self_judge": self_judge, "external_judge": external, "verifier": descriptive,
                     "splits": len(repeats)}


def experiment(args, directory, stages, started):
    progress = stages.progress
    deadline = started + LIMITS[args.phase]
    stages.start("preflight")
    stages.push(detail="claim_cert_preflight")
    prepare_dir = progress_root() / PREPARE_ID
    prepared = load(prepare_dir / PREPARE_SUMMARY)
    if prepared.get("verdict") != "READY":
        raise ValueError(f"prepare verdict is {prepared.get('verdict')}")
    corpus_path = Path(args.corpus).resolve(strict=True)
    if str(corpus_path) != prepared["corpus"]["path"] or corpus_path.stat().st_size != prepared["corpus"]["bytes"]:
        raise ValueError("--corpus differs from the prepared corpus")
    source = progress_root() / args.source_run
    source_summary = load(source / SOURCE_SUMMARY)
    source_inputs = load(source / "inputs.json")
    if source_summary.get("phase") != "run" or source_inputs["provenance"].get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"{args.source_run} is not a complete answer-prior run")
    knownness = load(source / "knownness.json")
    questions = load(prepare_dir / "questions.json")
    claims = build_claims(questions, knownness, args.phase)
    models, model_sha256 = {}, {}
    for name, path in (("verifier", args.verifier), ("e5", args.e5), ("judge", args.judge)):
        if path is not None:
            models[name], model_sha256[name] = model_inputs(path)
    found = versions()
    if found["vllm"] is None or found["torch"] is None:
        raise RuntimeError(f"vllm and torch are required; found {found}")
    shards = stored_shards(args.index_root)
    inputs = {"protocol": PROTOCOL, "repo_revision": progress.commit, "model_sha256": model_sha256, "models": models,
              "corpus_range_sha256": prepared["corpus"]["range_sha256"], "source_run": args.source_run,
              "source_generator_sha256": source_inputs["provenance"]["model_sha256"].get("generator"),
              "verifier_is_generator": model_sha256["verifier"] ==
              source_inputs["provenance"]["model_sha256"].get("generator"),
              "versions": found, "judge_tp": args.judge_tp if args.judge else None,
              "batching": {"gpu_memory": args.gpu_memory, "max_num_seqs": args.max_num_seqs,
                           "max_batched_tokens": args.max_batched_tokens, "chunk": args.chunk},
              "index_shards": [shard["manifest"] for shard in shards]}
    if args.phase == "run":
        smoke = load(progress_root() / args.smoke_run / SUMMARY)
        keys = ("protocol", "repo_revision", "model_sha256", "corpus_range_sha256", "source_run", "versions")
        if smoke.get("phase") != "smoke" or smoke.get("verdict") != "PASS" or \
                {key: smoke["provenance"].get(key) for key in keys} != {key: inputs[key] for key in keys}:
            raise ValueError("run requires a PASS smoke from this commit with the same models, corpus and source run")
    gpus = visible_gpus(args.gpus)
    if args.judge and args.judge_tp > len(gpus):
        raise ValueError("--judge-tp exceeds the visible GPUs")
    atomic_json(directory / "inputs.json", {"provenance": inputs, "phase": args.phase, "claims": len(claims),
                                           "gpus": gpus})
    context = {"directory": directory, "gpus": gpus, "deadline": deadline, "stages": stages, "bm25": None,
               "models": models, "model_sha256": model_sha256, "corpus": str(corpus_path),
               "offsets": str(prepare_dir / "offsets.npy"), "corpus_sha256": prepared["corpus"]["range_sha256"],
               "gpu_memory": args.gpu_memory, "max_num_seqs": args.max_num_seqs,
               "max_batched_tokens": args.max_batched_tokens, "chunk": args.chunk}
    stages.finish("preflight", note=f"{len(claims):,} 条论断 · 真 {sum(c['truth'] for c in claims):,}")
    search_stage(context, claims, shards)
    verified, verify_meta = verify_stage(context, claims, "verify", models["verifier"], 1, CONFIGS)
    judged, judge_meta = None, None
    if args.judge:
        judged, judge_meta = verify_stage(context, claims, "judge", models["judge"], args.judge_tp, ("q20",))
    else:
        stages.finish("judge", state="skipped", note="未提供 --judge，判官误差只用自判官")
    stages.start("analysis")
    stages.push(detail="claim_cert_analyze", metrics={"active_gpus": 0})
    import numpy as np

    search = np.load(directory / "search.npz")
    validity = {
        "claims": len(claims) >= MIN[args.phase]["claims"],
        "true_claims": sum(claim["truth"] for claim in claims) >= MIN[args.phase]["true"],
        "parsed": all(value >= MIN[args.phase]["parsed"] for value in verify_meta["parsed"].values()),
        "search_filled": bool((search["q_ids"] >= 0).all() and (search["qa_ids"] >= 0).all()),
    }
    records, result = analysis(context, claims, verified, judged, args.phase)
    validity["verifier_auc"] = (result["verifier"]["q20"]["auc"] or 0.0) >= AUC_FLOOR
    decision = decide(result["g1"], result["self_judge"], validity)
    with (directory / "claims.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {"claims": len(claims), "true": sum(claim["truth"] for claim in claims),
              "by_stratum": {label: sum(claim["stratum"] == label for claim in claims)
                             for label in ("known", "partial", "wrong", "unknown")}}
    summary = {**decision, **result, "validity": validity, "counts": counts, "verify": verify_meta,
               "judge_run": judge_meta, "provenance": inputs, "gpus": gpus, "alphas": list(ALPHAS)}
    if args.phase == "smoke":
        summary["run_verdict_preview"] = summary.pop("verdict")
        checks = {"stages_complete": True, **validity}
        summary["smoke_checks"] = checks
        summary["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    stages.finish("analysis", note=summary["verdict"])
    return summary


def run(args):
    if os.name != "posix":
        raise RuntimeError("run on the factory Linux environment")
    started = monotonic()
    directory = progress_root() / args.id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    progress = Progress(directory, PHASE)
    stages = Stages(progress, STAGE_PLAN[args.phase])
    signal.signal(signal.SIGTERM, stop)
    try:
        with progress.heartbeat():
            summary = experiment(args, directory, stages, started)
            summary.update(phase=args.phase, wall_seconds=round(monotonic() - started, 1))
            atomic_json(directory / SUMMARY, summary)
            verdict = summary["verdict"].lower().replace("-", "_")
            progress.update("done" if verdict in ("pass", "go") else "blocked", len(stages.rows), len(stages.rows),
                            detail=f"claim_cert_{verdict}", metrics={"active_gpus": 0}, stages=stages.snapshot())
            g1 = [row for row in summary.get("g1", []) if row["alpha"] == 0.1]
            print(json.dumps({"phase": args.phase, "verdict": summary["verdict"],
                              "gain_at_alpha_0.1": {row["target"]: row["mean_gain"] for row in g1},
                              "wall_seconds": summary["wall_seconds"]}, ensure_ascii=False, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(directory / "incomplete.json", {"verdict": "INCOMPLETE", "phase": args.phase,
                                                    "repo_revision": progress.commit, "error": str(exc)})
        stages.fail_running()
        progress.update("failed", detail="claim_cert_incomplete", stages=stages.snapshot())
        raise


def main():
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "run", "worker"))
    parser.add_argument("--id")
    parser.add_argument("--source-run", default="answer-prior-run-v1", help="answer-prior run with knownness.json")
    parser.add_argument("--corpus", type=Path, help="wiki-18.jsonl (the prepared corpus)")
    parser.add_argument("--verifier", type=Path, help="Qwen3-4B (thinking switch off)")
    parser.add_argument("--e5", type=Path, help="intfloat/e5-base-v2 (the model of the stored shards)")
    parser.add_argument("--index-root", type=Path, help="stored corpus vectors from answer-prior-run-v1")
    parser.add_argument("--judge", type=Path, help="optional stronger judge model, run on q20 only")
    parser.add_argument("--judge-tp", type=int, default=1, help="tensor-parallel GPUs per judge replica")
    parser.add_argument("--smoke-run")
    parser.add_argument("--gpus", type=int, choices=range(1, MAX_GPUS + 1), default=MAX_GPUS)
    parser.add_argument("--gpu-memory", type=float, default=0.92)
    parser.add_argument("--max-num-seqs", type=int, default=1024)
    parser.add_argument("--max-batched-tokens", type=int, default=131072)
    parser.add_argument("--chunk", type=int, default=1024, help="claims per vLLM call (x8 prompts)")
    parser.add_argument("--kind", choices=("search", "verify", "judge"), help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "worker":
        worker_main(args)
        return
    if not RUN_ID.fullmatch(args.id or ""):
        parser.error("--id must be claim-cert-<name> (lowercase letters, digits, dashes)")
    if None in (args.corpus, args.verifier, args.e5, args.index_root):
        parser.error(f"{args.phase} requires --corpus, --verifier, --e5 and --index-root")
    if args.phase == "run" and not RUN_ID.fullmatch(args.smoke_run or ""):
        parser.error("run requires --smoke-run claim-cert-<name> of a PASS smoke")
    if not 0.3 <= args.gpu_memory <= 0.97:
        parser.error("--gpu-memory must be within 0.3..0.97")
    run(args)


if __name__ == "__main__":
    main()
