"""Sanity check of a Dr. Zero solver training pool (P0 validity item).

Upstream `main_generation.py` writes questions back with chained pandas
assignment (`dataset.iloc[idx].prompt = message`). That only works while the
row is a view (pandas < 3 without copy-on-write). If it silently fails, the
solver would train on proposer prompts. This check catches that.

Usage:
    python -m work3.pool_check POOL.parquet [POOL2.parquet ...] [--out pool_check.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHALLENGER_MARKER = "You are an expert in question generation"
SOLVER_MARKER = "Answer the given question."


def _content(prompt) -> str:
    msgs = list(prompt)
    if not msgs:
        return ""
    first = msgs[0]
    return str(first.get("content", "")) if hasattr(first, "get") else str(first)


def _targets(reward_model) -> list:
    try:
        target = reward_model["ground_truth"]["target"]
    except (KeyError, TypeError):
        return []
    if target is None:
        return []
    return [str(x) for x in list(target) if x is not None]


def check_rows(rows) -> dict:
    """rows: iterable of (prompt, reward_model, data_source)."""
    n = solver = challenger = empty_answer = 0
    hops: dict[str, int] = {}
    for prompt, reward_model, data_source in rows:
        n += 1
        content = _content(prompt)
        if CHALLENGER_MARKER in content:
            challenger += 1
        elif content.lstrip().startswith(SOLVER_MARKER):
            solver += 1
        targets = _targets(reward_model)
        if not targets or not any(t.strip() for t in targets):
            empty_answer += 1
        hops[str(data_source)] = hops.get(str(data_source), 0) + 1
    ok = n > 0 and challenger == 0 and solver == n
    return {
        "rows": n,
        "solver_prompts": solver,
        "challenger_prompts": challenger,
        "empty_answers": empty_answer,
        "data_sources": dict(sorted(hops.items())),
        "ok": ok,
    }


def check_file(path: str | Path) -> dict:
    import pandas as pd

    df = pd.read_parquet(path)
    result = check_rows(zip(df["prompt"], df["reward_model"], df["data_source"]))
    result["file"] = str(path)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    files = [check_file(p) for p in args.pools]
    result = {"ok": all(f["ok"] for f in files), "files": files}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
