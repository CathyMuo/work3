"""P0 behaviour diagnosis from the evaluation dumps (CPU only, a few minutes).

    python -m work3.p0_diag --eval-root DRZERO_DIR/work3_p0/eval --out DRZERO_DIR/work3_p0/p0_diag.json

Exploratory, not a gate. Factory feedback #10: from iteration 1 to 3 single-hop EM rose while
multi-hop EM fell, although each pool keeps the 4:3:2:1 hop split (that split is fixed by the
seed file, so it says nothing about the questions the proposer actually wrote). This reads the
solver's own trajectories to see whether its *search behaviour* changed:

- per dataset and iteration: questions, EM, valid searches per question, share with no
  search, share without an <answer>, mean response length (characters);
- EM by number of searches (0, 1, 2, 3+) per dataset group;
- paired changes between iterations on the same question (the dumps are written in test-set
  order; questions are matched by data source and prompt text): correct->wrong,
  wrong->correct, exact McNemar p-value, and the mean change in searches.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from work3.p0_gate import DATASETS, SOURCE_PREFIX, resolve_dump

SINGLE = ("nq", "triviaqa", "popqa")
MULTI = ("hotpotqa", "2wikimultihopqa", "musique", "bamboogle")
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def valid_searches(text: str) -> int:
    """Same rule as Dr. Zero's count_valid_tool_calls."""
    n = 0
    for m in TOOL_CALL.findall(text):
        try:
            p = json.loads(m.strip())
        except ValueError:
            continue
        if (isinstance(p, dict) and p.get("name") == "search" and isinstance(p.get("arguments"), dict)
                and isinstance(p["arguments"].get("query_list"), list)):
            n += 1
    return n


def load(dump_dir: Path) -> Dict[tuple, dict]:
    rows: Dict[tuple, dict] = {}
    with resolve_dump(dump_dir).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = str(r.get("data_source", ""))
            ds = src[len(SOURCE_PREFIX):] if src.startswith(SOURCE_PREFIX) else src
            out = str(r.get("output", ""))
            key = (ds, str(r.get("input", "")))
            rows[key] = None if key in rows else {      # duplicated prompts cannot be paired
                "ds": ds, "correct": float(r["score"]) == 1.0, "searches": valid_searches(out),
                "answered": bool(ANSWER.search(out)), "chars": len(out)}
    return {k: v for k, v in rows.items() if v is not None}


def mcnemar_p(b: int, c: int) -> Optional[float]:
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return None
    lo = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(lo + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2 * tail)


def describe(rows: List[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {"n": n, "em": sum(r["correct"] for r in rows) / n,
            "searches_mean": sum(r["searches"] for r in rows) / n,
            "no_search_share": sum(r["searches"] == 0 for r in rows) / n,
            "no_answer_share": sum(not r["answered"] for r in rows) / n,
            "chars_mean": sum(r["chars"] for r in rows) / n}


def by_searches(rows: List[dict]) -> dict:
    out = {}
    for label, test in (("0", lambda s: s == 0), ("1", lambda s: s == 1), ("2", lambda s: s == 2),
                        ("3+", lambda s: s >= 3)):
        sel = [r for r in rows if test(r["searches"])]
        out[label] = {"share": len(sel) / len(rows) if rows else None,
                      "em": (sum(r["correct"] for r in sel) / len(sel)) if sel else None}
    return out


def paired(a: Dict[tuple, dict], b: Dict[tuple, dict], names) -> dict:
    keys = [k for k in a if k in b and k[0] in names]
    cw = sum(a[k]["correct"] and not b[k]["correct"] for k in keys)
    wc = sum(not a[k]["correct"] and b[k]["correct"] for k in keys)
    return {"pairs": len(keys), "correct_to_wrong": cw, "wrong_to_correct": wc,
            "em_change": (wc - cw) / len(keys) if keys else None, "mcnemar_p": mcnemar_p(cw, wc),
            "searches_change_mean": (sum(b[k]["searches"] - a[k]["searches"] for k in keys) / len(keys)
                                     if keys else None)}


def diagnose(eval_root: Path) -> dict:
    its = {}
    for t in (0, 1, 2, 3):
        d = eval_root / f"iter{t}"
        if d.is_dir():
            try:
                its[t] = load(d)
            except (OSError, ValueError) as e:
                print(f"[p0_diag] iter{t}: skipped ({e})", file=sys.stderr)
    groups = {**{d: (d,) for d in DATASETS}, "single_hop": SINGLE, "multi_hop": MULTI}
    res: dict = {"exploratory": True, "iterations": sorted(its), "per_iteration": {}, "paired": {}}
    for t, rows in its.items():
        vals = list(rows.values())
        res["per_iteration"][str(t)] = {
            g: {**describe([r for r in vals if r["ds"] in names]),
                **({"em_by_searches": by_searches([r for r in vals if r["ds"] in names])}
                   if g in ("single_hop", "multi_hop") else {})}
            for g, names in groups.items()}
    for a, b in ((0, 1), (1, 2), (2, 3), (1, 3)):
        if a in its and b in its:
            res["paired"][f"{a}->{b}"] = {g: paired(its[a], its[b], names) for g, names in groups.items()}
    return res


def table(res: dict) -> str:
    lines = ["group             iter     n     EM  search/q  no-search  no-answer   chars"]
    for g in (*DATASETS, "single_hop", "multi_hop"):
        for t in res["iterations"]:
            s = res["per_iteration"][str(t)].get(g, {})
            if s.get("n"):
                lines.append(f"{g:17s} {t:>4d} {s['n']:5d}  {s['em']:.3f}  {s['searches_mean']:8.2f}"
                             f"  {s['no_search_share']:9.3f}  {s['no_answer_share']:9.3f}  {s['chars_mean']:6.0f}")
    for k, v in res["paired"].items():
        for g in ("single_hop", "multi_hop"):
            p = v[g]
            if p["pairs"]:
                lines.append(f"paired {k} {g:10s}: pairs {p['pairs']}, correct->wrong {p['correct_to_wrong']}, "
                             f"wrong->correct {p['wrong_to_correct']}, McNemar p {p['mcnemar_p']:.2g}, "
                             f"searches change {p['searches_change_mean']:+.2f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = diagnose(Path(a.eval_root))
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
