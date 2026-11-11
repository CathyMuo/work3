"""Is a P0 stage's output complete? Used by run_p0.sh when a stage exits nonzero (factory
feedback #7, #8: the output was fully written, then the process was killed while shutting
down) and when deciding whether a finished stage can be skipped on a re-run.

    python -m work3.stage_check [--since EPOCH] ckpt <ckpt_dir> <final_step> <ngpu>
    python -m work3.stage_check [--since EPOCH] pool <pool.parquet>
    python -m work3.stage_check [--since EPOCH] pool_check <pool_check_iterN.json>
    python -m work3.stage_check [--since EPOCH] hf <solver_iterN_hf>
    python -m work3.stage_check [--since EPOCH] eval <eval/iterN> <test.parquet>

Exit 0 = complete. Otherwise the problems are printed and the exit code is 1.
--since: every checked output file must have been written at or after this time (the stage's
start), so a stage that dies early can never pass on an older run's output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def _fresh(paths, since: Optional[float], problems: List[str]) -> None:
    if since is None:
        return
    for p in paths:
        try:
            if os.path.getmtime(p) + 2 < since:
                problems.append(f"{p} predates this stage (written before it started)")
        except OSError:
            pass


def _nonempty(p: Path, problems: List[str]) -> bool:
    if not p.is_file() or p.stat().st_size == 0:
        problems.append(f"missing or empty {p}")
        return False
    return True


def check_ckpt(d: str, step: str, ngpu: str, since: Optional[float] = None) -> List[str]:
    """verl writes every rank's shards, then data.pt, then the tracker file last."""
    root, step, ngpu = Path(d), int(step), int(ngpu)
    g = root / f"global_step_{step}"
    actor = g / "actor"
    problems: List[str] = []
    tracker = root / "latest_checkpointed_iteration.txt"
    try:
        latest = tracker.read_text().strip()
        if latest != str(step):
            problems.append(f"{tracker.name} says {latest!r}, expected {step}")
    except OSError:
        problems.append(f"{tracker.name} missing")
    for kind in ("model", "optim", "extra_state"):
        for r in range(ngpu):
            _nonempty(actor / f"{kind}_world_size_{ngpu}_rank_{r}.pt", problems)
    for f in (g / "data.pt", actor / "huggingface" / "config.json"):
        if not f.is_file():
            problems.append(f"missing {f}")
    _fresh([tracker], since, problems)
    return problems


def check_pool(path: str, since: Optional[float] = None) -> List[str]:
    """The pool is written in one go at the very end (dataset.to_parquet); a kill during that
    write leaves a file pyarrow cannot read."""
    p = Path(path)
    if not _nonempty(p, problems := []):
        return problems
    try:
        import pandas as pd
        df = pd.read_parquet(p)
    except Exception as e:  # truncated / corrupt parquet
        return [f"{p} unreadable: {type(e).__name__}: {e}"]
    if len(df) == 0:
        problems.append(f"{p} has no rows")
    for col in ("prompt", "data_source", "reward_model"):
        if col not in df.columns:
            problems.append(f"{p} has no column {col}")
        elif df[col].isna().any():
            problems.append(f"{p}: {int(df[col].isna().sum())} empty {col} values")
    _fresh([p], since, problems)
    return problems


def check_pool_check(path: str, since: Optional[float] = None) -> List[str]:
    """Complete AND passed: a pool that fails its check must still stop the run."""
    p = Path(path)
    try:
        result = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{p} unreadable: {e}"]
    problems = [] if result.get("ok") is True else [f"{p}: pool check did not pass (ok={result.get('ok')!r})"]
    _fresh([p], since, problems)
    return problems


def check_hf(d: str, since: Optional[float] = None) -> List[str]:
    root = Path(d)
    problems: List[str] = []
    _nonempty(root / "config.json", problems)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        except (OSError, ValueError, KeyError) as e:
            return problems + [f"{index} unreadable: {e}"]
        for s in shards:
            _nonempty(root / s, problems)
        weights = [root / s for s in shards]
    else:
        weights = sorted(root.glob("model*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
        if not weights:
            problems.append(f"no model weights in {root}")
        for w in weights:
            _nonempty(w, problems)
    _fresh([root / "config.json"] + weights, since, problems)
    return problems


def check_eval(d: str, test_parquet: str, since: Optional[float] = None) -> List[str]:
    """The validation dump must parse line by line and hold every test question of every
    data source (the counts of the test parquet that was fed in)."""
    from work3.p0_gate import resolve_dump

    try:
        dump = resolve_dump(d)
    except (OSError, ValueError) as e:
        return [str(e)]
    counts: dict = {}
    try:
        with dump.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    row = json.loads(line)
                    float(row["score"])
                    s = str(row.get("data_source", ""))
                    counts[s] = counts.get(s, 0) + 1
    except (ValueError, KeyError, TypeError) as e:
        return [f"{dump} line {n} unreadable (cut off?): {type(e).__name__}: {e}"]
    try:
        import pandas as pd
        expected = pd.read_parquet(test_parquet, columns=["data_source"])["data_source"].astype(str).value_counts().to_dict()
    except Exception as e:
        return [f"cannot read expected counts from {test_parquet}: {e}"]
    problems = [f"{dump}: {s} has {counts.get(s, 0)} of {n} rows"
                for s, n in sorted(expected.items()) if counts.get(s, 0) != n]
    extra = sorted(set(counts) - set(expected))
    if extra:
        problems.append(f"{dump}: unexpected data sources {extra}")
    _fresh([dump], since, problems)
    return problems


CHECKS = {"ckpt": (check_ckpt, 3), "pool": (check_pool, 1), "pool_check": (check_pool_check, 1),
          "hf": (check_hf, 1), "eval": (check_eval, 2)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", type=float)
    ap.add_argument("kind", choices=sorted(CHECKS))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args(argv)
    fn, nargs = CHECKS[a.kind]
    if len(a.args) != nargs:
        ap.error(f"{a.kind} takes {nargs} arguments")
    problems = fn(*a.args, since=a.since)
    for line in problems[:8]:
        print(f"[p0 check] {a.kind}: {line}", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
