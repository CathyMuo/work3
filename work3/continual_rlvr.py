"""Factory launch validation and frozen-run admission for continual RLVR."""

import argparse
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from work3.continual_rlvr_data import load_bundle, sha256_file
from work3.continual_rlvr_protocol import BUDGET, PROTOCOL, check_scores, necessity_verdict
from work3.wiki_rsi_progress import REPO, atomic_json, progress_root

DEPENDENCIES = {"transformers": "4.56.2", "peft": "0.17.1"}


def identify_inputs(args):
    bundle = load_bundle(args.bundle)
    model = args.model.resolve(strict=True)
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) != ("qwen2", 2048, 36):
        raise ValueError("expected the frozen Qwen2.5-3B-Instruct architecture")
    weights = sorted(model.glob("model*.safetensors"))
    if not weights:
        raise ValueError("missing local safetensors weights")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if names != {path.name for path in weights}:
            raise ValueError("safetensors index and local shards differ")
    filenames = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
    files = [model / name for name in filenames]
    files.extend(path for path in (index, model / "special_tokens_map.json",
                                   model / "vocab.json", model / "merges.txt") if path.is_file())
    versions = {name: importlib.metadata.version(name) for name in ("torch", *DEPENDENCIES)}
    if any(versions[name] != version for name, version in DEPENDENCIES.items()):
        raise ValueError(f"expected {DEPENDENCIES}, found {versions}")
    return {
        "protocol": PROTOCOL, "bundle_sha256": sha256_file(args.bundle), "sources": bundle["sources"],
        "model_sha256": {path.name: sha256_file(path) for path in files + weights},
        "versions": versions, "python": sys.version.split()[0],
        "repo_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }, bundle


def admit_method(necessity, inputs, splits):
    with (necessity / "wiki_rsi_continual_necessity_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("provenance") != inputs:
        raise ValueError("C1 model, bundle, repository or runtime does not match this C2 run")
    raw = necessity / "necessity_rollouts.json"
    if sha256_file(raw) != summary["rollouts_sha256"]:
        raise ValueError("C1 rollouts changed")
    rows = json.loads(raw.read_text(encoding="utf-8"))
    for snapshot in ("before", "after"):
        check_scores(rows[snapshot]["a"], splits["a_dev"], 4)
        check_scores(rows[snapshot]["b"], splits["b_dev"], 32)
    if summary.get("verdict") != "GO" or necessity_verdict(rows["before"], rows["after"])["verdict"] != "GO":
        raise ValueError("frozen C1 gate did not pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "smoke", *BUDGET))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--id")
    parser.add_argument("--necessity-run")
    args = parser.parse_args()
    if args.phase != "preflight" and not re.fullmatch(r"continual-[a-z0-9][a-z0-9-]{0,55}", args.id or ""):
        parser.error("--id must be a fresh continual-<name> run id")
    if args.phase == "method" and not re.fullmatch(r"continual-[a-z0-9][a-z0-9-]{0,55}", args.necessity_run or ""):
        parser.error("method requires --necessity-run continual-<name>")
    if args.phase != "method" and args.necessity_run:
        parser.error("--necessity-run is for method only")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    inputs, bundle = identify_inputs(args)
    if args.phase == "preflight":
        print(json.dumps(inputs, ensure_ascii=False, indent=2))
        return
    root = progress_root()
    if args.phase == "method":
        admit_method(root / args.necessity_run, inputs, bundle["splits"])
    root.mkdir(parents=True, exist_ok=True)
    run = root / args.id
    run.mkdir()
    atomic_json(run / "inputs.json", inputs)
    if args.phase == "method":
        atomic_json(run / "admission.json", {"necessity_run": args.necessity_run,
                    "summary_sha256": sha256_file(root / args.necessity_run /
                                                 "wiki_rsi_continual_necessity_summary.json")})
    limits = {**BUDGET, "smoke": 600}
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
        "--max-restarts=0", "--module", "work3.continual_rlvr_run", args.phase, str(run),
        str(args.model.resolve()), str(args.bundle.resolve()), str(limits[args.phase]),
    ]
    from work3.wiki_rsi_progress import Progress

    try:
        with subprocess.Popen(command, cwd=REPO, start_new_session=True) as process:
            try:
                status = process.wait(timeout=limits[args.phase])
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # Let torchrun propagate shutdown to its own worker process groups.
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                raise
        if status:
            raise RuntimeError(f"torchrun exited with status {status}")
    except (OSError, subprocess.SubprocessError, RuntimeError, KeyboardInterrupt) as exc:
        progress = Progress(run, "continual_" + args.phase)
        progress.update("failed", detail="continual_incomplete")
        atomic_json(run / "incomplete.json", {"verdict": "INCOMPLETE", "error": str(exc)})
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
