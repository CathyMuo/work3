"""Run the frozen Work3 supervision-allocation experiment and publish local progress."""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from work3 import wiki_rsi_feedback_protocol as p
from work3.evidence_ref import sha256
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import Progress, atomic_json, progress_root
from work3.wiki_rsi_references import WIKI_RSI_REFERENCES

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "research/protocols/RSI_wikipedia_feedback_allocation_v2.md",
    ROOT / "research/protocols/RSI_wikipedia_feedback_allocation.md",
    *(ROOT / "work3" / f"wiki_rsi_feedback{suffix}.py" for suffix in ("", "_protocol", "_train", "_data")),
    *(ROOT / "work3" / f"{name}.py" for name in (
        "wiki_rsi_w2", "wiki_rsi_replay", "wiki_rsi_two_query_live", "wiki_rsi_natural_query",
        "wiki_rsi_action_model", "wiki_rsi_switch_train", "wiki_rsi_action_pilot", "evidence_ref",
    )),
)
RUN_ID = re.compile(r"feedback-[a-zA-Z0-9_-]{1,52}")


def receipt_path(folder, phase):
    return folder / f"wiki_rsi_feedback_{phase}_summary.json"


def receipt(folder, phase, contract):
    import json

    path = receipt_path(folder, phase)
    if not path.is_file():
        raise ValueError(f"Finish {phase} before starting the next phase")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["protocol"] != p.PROTOCOL or result["contract"] != contract or result["verdict"] != "GO":
        raise ValueError(f"{phase} did not pass under this frozen implementation")
    return result


def read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def telemetry(folder):
    path = folder / "worker_progress.json"
    if not path.is_file():
        return {}
    return read_json(path)


def run_worker(path):
    from work3 import wiki_rsi_feedback_data as data
    from work3 import wiki_rsi_feedback_train as train
    from work3 import wiki_rsi_action_model as model_tools
    from work3 import wiki_rsi_switch_train as training

    spec = read_json(path)
    out, experiment = path.parent, path.parent.parent.parent
    locked = experiment / data.LOCKED
    if sha256(locked) != spec["locked_sha256"]:
        raise ValueError("Frozen input file changed")
    inputs = read_json(locked)
    if spec["phase"] in {"mechanism", "pilot"}:
        pairs = experiment / data.PAIRS
        if sha256(pairs) != spec["pairs_sha256"]:
            raise ValueError("Paired training data changed")
        rows = read_json(pairs)["pairs"]
    else:
        rows = ()
    if spec["phase"] == "confirm":
        spec["adapter_path"] = str(experiment / spec["pilot_jobs_dir"] / spec["name"] / "adapter.pt")

    def progress(detail, done=0, total=0, **metrics):
        atomic_json(out / "worker_progress.json", {
            "detail": detail, "done": done, "total": total, "updated_at": int(time.time()), "metrics": metrics,
        })

    progress("feedback_loading")
    tok, model, device = training.load_solver(spec["model"])
    if spec["phase"] == "prepare":
        report = data.prepare_shard(spec, inputs, tok, model, device, model_tools.generate_queries, progress)
        report["pairs"] = [row for row in report["pairs"] if train.encode_pair(tok, row) is not None]
        report["funnel"]["encoded"] = len(report["pairs"])
    else:
        report = train.run(spec, inputs, rows, tok, model, device, progress, out)
    atomic_json(out / "result.json", report)
    return 0


def run_workers(folder, phase, specs, devices, deadline, progress, detail="feedback_workers"):
    root = Path(tempfile.mkdtemp(prefix=f"{phase}-workers-", dir=folder))
    queued, active, completed = list(specs), {}, {}
    errors = False
    try:
        while queued or active:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{phase} exhausted the frozen time budget")
            for device, (spec, child) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"Worker {spec['name']} exited {code}; consult its local log")
                completed[spec["name"]] = read_json(root / spec["name"] / "result.json")
                del active[device]
            for device in devices:
                if device in active or not queued:
                    continue
                spec = queued.pop(0)
                work = root / spec["name"]
                work.mkdir()
                config = work / "job.json"
                atomic_json(config, spec)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": device, "OMP_NUM_THREADS": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                with (work / "worker.log").open("wb") as logs:
                    child = subprocess.Popen([sys.executable, "-m", "work3.wiki_rsi_feedback", "_worker", str(config)],
                                             cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=logs, stderr=subprocess.STDOUT)
                active[device] = (spec, child)
            workers = [{"name": spec["name"], "device": device, **telemetry(root / spec["name"])}
                       for device, (spec, _) in active.items()]
            progress.update("running", len(completed), len(specs), detail=detail,
                            metrics={"active_gpus": len(active)}, workers=workers)
            if active:
                time.sleep(1)
    except BaseException:
        errors = True
        raise
    finally:
        if errors:
            for _, child in active.values():
                if child.poll() is None:
                    child.terminate()
            for _, child in active.values():
                try:
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    return root.name, completed


def verify_context(args, locked, needed):
    from work3 import wiki_rsi_replay as replay
    from work3 import wiki_rsi_w2 as wiki

    if locked["retriever_id"] != args.retriever_id:
        raise ValueError("Use the frozen retriever ID")
    parity, result = wiki.check_anchors(wiki.Retriever(args.retriever), locked["anchors"])
    if not parity:
        raise ValueError("The current retriever does not reproduce the frozen index anchors")
    for state, path in needed.items():
        if replay.model_fingerprint(path) != locked["models"][state]:
            raise ValueError(f"{state} checkpoint differs from the frozen model")
    return result


def reuse_v1_lock(args, folder, progress, versions):
    from work3 import wiki_rsi_feedback_data as data

    source = args.import_v1.resolve()
    lock_report, prep_report = (read_json(receipt_path(source, phase)) for phase in ("lock", "prepare"))
    if (source == folder.resolve() or
            any(rep["protocol"] != p.LEGACY_PROTOCOL or rep["repo_revision"] != p.LEGACY_RELEASE
                for rep in (lock_report, prep_report)) or
            lock_report["phase"] != "lock" or prep_report["phase"] != "prepare" or
            lock_report["verdict"] != "GO" or prep_report["verdict"] != "NO-GO" or prep_report["count"] != 0 or
            prep_report["pairs_sha256"] is not None or lock_report["contract"] != prep_report["contract"] or
            any(rep["runtime_versions"] != versions for rep in (lock_report, prep_report))):
        raise ValueError("Reuse requires the original v1 lock GO and zero-pair prepare NO-GO")
    old_path = source / data.LOCKED
    old_digest = sha256(old_path)
    if old_digest != lock_report["locked_sha256"]:
        raise ValueError("The archived v1 locked input has changed")
    locked = read_json(old_path)
    titles = locked["titles"]
    if (locked["protocol"] != p.LEGACY_PROTOCOL or len(titles) != p.PROPOSALS or
            len(set(titles)) != len(titles) or locked["source"] != lock_report["inputs"] or
            locked["models"] != lock_report["models"]):
        raise ValueError("The v1 lock artifact does not match its recorded input and models")
    progress.update("running", detail="feedback_import_v1")
    anchors = verify_context(args, locked, {"early": args.early_model, "late": args.late_model})
    atomic_json(folder / data.LOCKED, {**locked, "protocol": p.PROTOCOL})
    return {"ready": True, "locked_sha256": sha256(folder / data.LOCKED), "v1_locked_sha256": old_digest,
            "v1_receipts_sha256": {phase: sha256(receipt_path(source, phase)) for phase in ("lock", "prepare")},
            **{key: lock_report[key] for key in ("by_split", "clusters", "sources", "inputs", "models",
                                                "drzero_revision")},
            "anchor_checks": anchors, "shared_first_queries": 0,
            "reused_first_query_cache": lock_report["shared_first_queries"]}


def prepare(folder, progress, deadline, locked, anchors, common, devices, model):
    from work3 import wiki_rsi_feedback_data as data

    reports, jobs, counts = {}, [], Counter()
    screen = {}
    for label, start, stop in (("screen", 0, p.SCREEN_SOURCES), ("tail", p.SCREEN_SOURCES, p.PROPOSALS)):
        progress.update("running", detail=f"feedback_prepare_{label}")
        specs = [{**common, "name": f"source-{label}-{index}", "shard": index, "model": str(model),
                  "source_range": [start, stop]} for index in range(4)]
        part_deadline = min(deadline, time.monotonic() + p.SCREEN_SECONDS) if label == "screen" else deadline
        directory, result = run_workers(folder, "prepare", specs, devices, part_deadline, progress,
                                        detail=f"feedback_prepare_{label}")
        jobs.append(directory)
        reports.update(result)
        rows = data.merge_pairs(reports.values(), locked)
        counts = Counter(row["family"] for row in rows)
        if label == "screen":
            screen = {"count": len(rows), "by_family": dict(counts), "passed": p.enough_screen_pairs(counts)}
            if not screen["passed"]:
                break
    ready = screen["passed"] and p.enough_pairs(counts)
    if ready:
        atomic_json(folder / data.PAIRS, {"protocol": p.PROTOCOL, "pairs": rows})
    funnel = Counter()
    for report in reports.values():
        funnel.update(report["funnel"])
    return {"ready": ready, "count": len(rows), "by_family": dict(counts), "anchor_checks": anchors,
            "reason": "ready" if ready else "insufficient_pairs" if screen["passed"] else "screen_no_go",
            "screen": screen, "sources": sum(row["sources"] for row in reports.values()),
            "triage": dict(Counter(row["triage"] for row in rows)),
            "question_first_witness": sum(row["question_first_witness"] for row in rows),
            "funnel": dict(funnel), "queries": sum(row["queries"] for row in reports.values()),
            "jobs_dirs": jobs, "pairs_sha256": sha256(folder / data.PAIRS) if ready else None}


def experiment(args, folder, progress, contract, deadline):
    from work3 import wiki_rsi_feedback_data as data

    phase = args.phase
    if phase == "lock":
        return data.lock(args, folder, progress, deadline)
    frozen = receipt(folder, "lock", contract)
    path = folder / data.LOCKED
    if sha256(path) != frozen["locked_sha256"]:
        raise ValueError("Frozen candidate/evaluation file changed")
    locked = read_json(path)
    devices = args.gpus.split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or any(not device.isdecimal() for device in devices):
        raise ValueError("Supply exactly four distinct local GPU IDs")
    model_paths = {"late": args.late_model.resolve()}
    if phase == "mechanism":
        model_paths["early"] = args.early_model.resolve()
    progress.update("running", detail="feedback_models")
    anchors = verify_context(args, locked, model_paths)
    common = {"phase": phase, "retriever": args.retriever, "locked_sha256": frozen["locked_sha256"]}
    if phase == "prepare":
        return prepare(folder, progress, deadline, locked, anchors, common, devices, model_paths["late"])

    if phase == "pilot":
        receipt(folder, "mechanism", contract)
    prior = receipt(folder, "prepare", contract)
    if sha256(folder / data.PAIRS) != prior["pairs_sha256"]:
        raise ValueError("Approved training pairs changed")
    common["pairs_sha256"] = prior["pairs_sha256"]
    fixed = None
    if phase == "mechanism":
        keys = [(seed, state, arm) for seed in p.SEEDS for state in ("early", "late") for arm in p.SIGNALS]
        specs = [{**common, "name": f"{seed}-{state}-{arm}", "seed": seed, "arm": arm,
                  "model": str(model_paths[state])} for seed, state, arm in keys]
    else:
        arms = p.ARMS
        if phase == "confirm":
            pilot = receipt(folder, "pilot", contract)
            fixed = pilot["chosen_fixed"]
            arms = ("adaptive", fixed, "triage", "frozen_probe")
        keys = [(seed, arm) for seed in p.SEEDS for arm in arms]
        keys.append((p.SEEDS[0], "base"))
        specs = [{**common, "name": f"{seed}-{arm}", "seed": seed, "arm": arm, "model": str(model_paths["late"])}
                 for seed, arm in keys]
        if phase == "confirm":
            for spec in specs:
                if spec["arm"] != "base":
                    spec.update(pilot["arms"][spec["name"]], pilot_jobs_dir=pilot["jobs_dir"])
    jobs, reports = run_workers(folder, phase, specs, devices, deadline, progress)
    grouped = {key: reports[spec["name"]] for key, spec in zip(keys, specs, strict=True)}
    split = "confirmation" if phase == "confirm" else "development"
    expected = {row["qid"] for row in locked["splits"][split]}
    if any(expected != row["scores"].keys() for row in grouped.values()):
        raise ValueError("Incomplete evaluation of the frozen split")
    if phase == "mechanism":
        decision = p.mechanism_gate(grouped)
    else:
        if phase == "pilot":
            if (any(grouped[seed, arm]["train_steps"] != p.ROUNDS * p.TRAIN_STEPS
                    for seed in p.SEEDS for arm in p.ARMS) or
                    any(len({grouped[seed, arm]["padded_train_tokens"] for arm in p.ARMS}) != 1
                        for seed in p.SEEDS)):
                raise ValueError("Matched training compute accounting failed")
        decision = p.end_to_end(grouped, fixed, final=phase == "confirm")
    decision.update(jobs_dir=jobs, anchor_checks=anchors, by_arm={
        name: {**{key: value for key, value in row.items() if key != "scores"},
               "correct": sum(case["correct"] for case in row["scores"].values()), "n": len(row["scores"])}
        for name, row in reports.items()})
    if phase == "pilot":
        decision["arms"] = {name: {key: row[key] for key in ("weights_sha256", "training_seconds", "choices")}
                            for name, row in reports.items() if "weights_sha256" in row}
    return decision


def main(argv=None):
    import json

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "_worker":
        return run_worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=p.PHASES)
    parser.add_argument("--run", required=True)
    parser.add_argument("--original-run", type=Path)
    parser.add_argument("--import-v1", type=Path, help="reuse the exact v1 frozen split in a fresh v2 run")
    parser.add_argument("--drzero", type=Path)
    parser.add_argument("--test-parquet", type=Path)
    parser.add_argument("--eval-dump", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--references", type=Path, default=WIKI_RSI_REFERENCES)
    parser.add_argument("--retriever", default=os.environ.get("WIKI_RSI_RETRIEVER", "http://127.0.0.1:8000/retrieve"))
    parser.add_argument("--retriever-id", default=os.environ.get("WIKI_RSI_RETRIEVER_ID", "p0-e5-wiki18"))
    parser.add_argument("--early-model", type=Path, default=os.environ.get("P0_ITER1_MODEL"))
    parser.add_argument("--late-model", type=Path, required=not os.environ.get("P0_ITER3_MODEL"),
                        default=os.environ.get("P0_ITER3_MODEL"))
    parser.add_argument("--gpus", default=os.environ.get("WIKI_RSI_FEEDBACK_GPUS", "0,1,2,3"))
    args = parser.parse_args(argv)
    if not RUN_ID.fullmatch(args.run):
        parser.error("Use a fresh feedback-... run ID")
    if args.import_v1 and args.phase != "lock":
        parser.error("--import-v1 is only accepted by lock")
    if args.phase == "lock" and not (args.early_model and (args.import_v1 or (args.original_run and args.drzero))):
        parser.error("lock requires --early-model and either --import-v1 or --original-run plus --drzero")
    if args.phase == "mechanism" and not args.early_model:
        parser.error("mechanism requires --early-model")

    os.umask(0o077)
    folder = progress_root() / args.run
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        parser.error("Use an ordinary local run directory")
    marker = folder / ".feedback_phase.lock"
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        parser.error("This run already has an active phase")
    try:
        destination = receipt_path(folder, args.phase)
        if destination.exists():
            parser.error("This phase already has a final decision; use its saved summary")
        if args.phase == "lock" and (folder / "wiki_rsi_feedback_prepare_summary.json").exists():
            parser.error("Use a fresh run ID to protect the prior experiment")
        contract = {path.name: sha256(path) for path in CONTRACT}
        versions = {"python": sys.version.split()[0], **{
            name: version(name) for name in ("torch", "transformers", "numpy", "pyarrow")}}
        if args.phase != "lock":
            try:
                index = p.PHASES.index(args.phase)
                previous = receipt(folder, p.PHASES[index - 1], contract)
                if previous["runtime_versions"] != versions:
                    raise ValueError("Use the same runtime versions as the preceding phase")
            except ValueError as error:
                parser.error(str(error))
        progress = Progress(folder, f"feedback_{args.phase}")
        progress.update("running", detail="feedback_setup")
        start = time.monotonic()
        try:
            with progress.heartbeat():
                result = (reuse_v1_lock(args, folder, progress, versions) if args.import_v1 else
                          experiment(args, folder, progress, contract, start + p.HOURS[args.phase] * 3600))
                if time.monotonic() - start >= p.HOURS[args.phase] * 3600:
                    raise TimeoutError("Frozen phase budget exhausted")
        except TimeoutError:
            result = {"ready": False, "reason": "time_budget_exhausted"}
        except (Exception, KeyboardInterrupt):
            progress.update("failed", detail="feedback_infra")
            raise
        ready = result.pop("ready")
        result.update(verdict="GO" if ready else "NO-GO", phase=args.phase, protocol=p.PROTOCOL,
                      repo_revision=revision(ROOT), contract=contract, runtime_versions=versions,
                      wall_seconds=round(time.monotonic() - start, 3))
        atomic_json(destination, result)
        current = progress.data["phases"][progress.phase]
        progress.update("done" if ready else "blocked", current["done"] or int(ready), current["total"] or 1,
                        detail=("feedback_go" if ready else {"time_budget_exhausted": "feedback_deadline",
                                                            "screen_no_go": "feedback_screen_no_go"}.get(
                                                                result.get("reason"), "feedback_no_go")),
                        metrics={"wall_seconds": result["wall_seconds"],
                                 **({"verified_pairs": result["count"]} if "count" in result else {})})
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0 if ready else 2
    finally:
        marker.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
