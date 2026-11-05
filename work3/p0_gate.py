"""P0 gate: Dr. Zero Qwen2.5-3B reproduction.

Decision rule frozen in research/protocols/P0_drzero_reproduction.md. Reads the
validation dumps written by verl (`<step>.jsonl`, one JSON object per sample with
`data_source` and `score`) and prints / writes a JSON summary.

Usage:
    python -m work3.p0_gate --iter3 DUMP [--iter0 DUMP] [--iter1 DUMP]
        [--iter2 DUMP] [--out p0_summary.json]

A DUMP may be a .jsonl file or a directory holding exactly one .jsonl file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS = (
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
)
SOURCE_PREFIX = "searchR1_"
# Full test / dev sizes used by Search-R1 (FlashRAG). Recorded, not gating.
FLASHRAG_COUNTS = {
    "nq": 3610,
    "triviaqa": 11313,
    "popqa": 14267,
    "hotpotqa": 7405,
    "2wikimultihopqa": 12576,
    "musique": 2417,
    "bamboogle": 125,
}
# arXiv 2601.07055 v2, Table 3 (Qwen2.5-3B-Instruct).
PAPER = {
    1: {"nq": 0.381, "triviaqa": 0.526, "popqa": 0.392, "hotpotqa": 0.284,
        "2wikimultihopqa": 0.243, "musique": 0.084, "bamboogle": 0.216},
    2: {"nq": 0.401, "triviaqa": 0.563, "popqa": 0.408, "hotpotqa": 0.289,
        "2wikimultihopqa": 0.255, "musique": 0.102, "bamboogle": 0.216},
    3: {"nq": 0.397, "triviaqa": 0.572, "popqa": 0.431, "hotpotqa": 0.298,
        "2wikimultihopqa": 0.291, "musique": 0.091, "bamboogle": 0.200},
}
PAPER_AVG_ITER3 = 0.326
TOLERANCE = 0.020
THRESHOLD = round(PAPER_AVG_ITER3 - TOLERANCE, 3)  # 0.306
EPS = 1e-12


def resolve_dump(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
        if len(files) != 1:
            raise ValueError(f"{p}: expected exactly one .jsonl file, found {len(files)}")
        return files[0]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def summarize_dump(path: str | Path) -> dict:
    """Per-dataset EM, counts and validity problems for one dump."""
    dump = resolve_dump(path)
    correct = {d: 0 for d in DATASETS}
    count = {d: 0 for d in DATASETS}
    other_sources: dict[str, int] = {}
    nonbinary = 0
    with dump.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = str(row.get("data_source", ""))
            name = source[len(SOURCE_PREFIX):] if source.startswith(SOURCE_PREFIX) else None
            if name not in count:
                other_sources[source] = other_sources.get(source, 0) + 1
                continue
            score = float(row["score"])
            if score not in (0.0, 1.0):
                nonbinary += 1
            count[name] += 1
            correct[name] += 1 if score == 1.0 else 0

    problems = []
    missing = [d for d in DATASETS if count[d] == 0]
    if missing:
        problems.append(f"missing datasets: {', '.join(missing)}")
    if nonbinary:
        problems.append(f"{nonbinary} non-binary scores")
    em = {d: (correct[d] / count[d] if count[d] else None) for d in DATASETS}
    average = None if missing else sum(em[d] for d in DATASETS) / len(DATASETS)
    return {
        "dump": str(dump),
        "em": em,
        "average": average,
        "counts": count,
        "counts_match_flashrag": all(count[d] == FLASHRAG_COUNTS[d] for d in DATASETS),
        "other_sources": other_sources,
        "problems": problems,
    }


def decide(iterations: dict[int, dict], pool_check_ok: bool | None = None) -> dict:
    """Apply the frozen P0 rule to per-iteration summaries."""
    problems = []
    if 3 not in iterations:
        problems.append("iteration-3 dump missing")
    for t, s in sorted(iterations.items()):
        for p in s["problems"]:
            problems.append(f"iter{t}: {p}")
    if pool_check_ok is False:
        problems.append("pool check failed")

    if problems:
        verdict = "INVALID"
    elif iterations[3]["average"] + EPS >= THRESHOLD:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    deltas = {}
    for t, s in iterations.items():
        if t in PAPER and s["average"] is not None:
            deltas[t] = {
                "average": s["average"] - sum(PAPER[t].values()) / len(DATASETS),
                **{d: s["em"][d] - PAPER[t][d] for d in DATASETS},
            }
    return {
        "protocol": "research/protocols/P0_drzero_reproduction.md",
        "threshold_iter3_average": THRESHOLD,
        "verdict": verdict,
        "problems": problems,
        "pool_check_ok": pool_check_ok,
        "iterations": {str(t): s for t, s in sorted(iterations.items())},
        "delta_vs_paper": {str(t): d for t, d in sorted(deltas.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for t in range(4):
        ap.add_argument(f"--iter{t}", help=f"validation dump for iteration {t}")
    ap.add_argument("--pool-check", help="JSON written by scripts/p0/check_pool.py")
    ap.add_argument("--out", help="write the summary JSON here")
    args = ap.parse_args(argv)

    iterations = {}
    for t in range(4):
        path = getattr(args, f"iter{t}")
        if path:
            iterations[t] = summarize_dump(path)
    pool_ok = None
    if args.pool_check:
        pool_ok = bool(json.loads(Path(args.pool_check).read_text(encoding="utf-8"))["ok"])
    result = decide(iterations, pool_ok)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
