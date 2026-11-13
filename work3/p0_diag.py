"""P0 behaviour diagnosis from the evaluation dumps (CPU only, a few minutes).

    python -m work3.p0_diag --eval-root OUT/eval --test-parquet DRZERO_DIR/data/test.parquet \\
        [--compare micro2=OUT/eval_ablate/micro2 ...] --out OUT/p0_diag.json

Exploratory, not a gate. Feedback #10/#11: from iteration 1 to 3, single-hop EM rose and
multi-hop EM fell, and the solver issued fewer search calls. This reads the solver's own
trajectories, per dataset (and single-/multi-hop group) and iteration:

- EM; search calls per question; queries per question and per call (one call may carry
  several queries, so fewer calls is not necessarily less retrieval); no-search and
  no-answer shares; response length in characters;
- with the gold answers from the test parquet: the share of questions whose gold answer
  appears in any retrieved passage, and the split of wrong answers into "gold never
  retrieved" (search stopped or went astray before reaching it) and "gold retrieved but
  not answered" (reading / reasoning failure);
- EM and gold-retrieval share by number of search calls (0, 1, 2, 3+);
- paired changes between iterations on the same question (matched by data source and
  question text; questions whose text repeats within a data source are left out of all
  statistics): correct->wrong, wrong->correct, exact McNemar p-value, mean change in
  searches, and the flips split by whether the search count fell, stayed or rose.

--compare NAME=DIR adds another evaluation dump (e.g. a retrained iteration-3 solver),
described the same way and paired against iterations 2 and 3.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional

from work3.p0_gate import DATASETS, SOURCE_PREFIX, resolve_dump

SINGLE = ("nq", "triviaqa", "popqa")
MULTI = ("hotpotqa", "2wikimultihopqa", "musique", "bamboogle")
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize(s: str) -> str:
    """Dr. Zero / Search-R1 answer normalisation."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def search_calls(text: str) -> List[int]:
    """Query count of every valid search call (same validity rule as Dr. Zero's
    count_valid_tool_calls)."""
    out = []
    for m in TOOL_CALL.findall(text):
        try:
            p = json.loads(m.strip())
        except ValueError:
            continue
        if (isinstance(p, dict) and p.get("name") == "search" and isinstance(p.get("arguments"), dict)
                and isinstance(p["arguments"].get("query_list"), list)):
            out.append(len(p["arguments"]["query_list"]))
    return out


def valid_searches(text: str) -> int:
    return len(search_calls(text))


def retrieved_texts(text: str) -> List[str]:
    out = []
    for m in TOOL_RESPONSE.findall(text):
        m = m.strip()
        try:
            m = str(json.loads(m).get("result", m))
        except (ValueError, AttributeError):
            pass
        out.append(normalize(m))
    return out


def question_of(prompt: str) -> str:
    """The solver prompt ends with "Question: {question}" (Dr. Zero DEFAULT_SOLVER_PREFIX); the
    decoded dump input may carry a trailing "assistant" role line."""
    i = prompt.rfind("Question: ")
    q = prompt[i + len("Question: "):] if i >= 0 else prompt
    q = re.sub(r"\s*\n\s*assistant\s*$", "", q)
    return q.strip()


def load_gold(test_parquet: Optional[str]) -> Dict[tuple, List[str]]:
    if not test_parquet:
        return {}
    import pandas as pd
    df = pd.read_parquet(test_parquet, columns=["data_source", "extra_info", "reward_model"])
    gold: Dict[tuple, List[str]] = {}
    for src, info, rm in zip(df["data_source"], df["extra_info"], df["reward_model"]):
        ds = str(src)[len(SOURCE_PREFIX):] if str(src).startswith(SOURCE_PREFIX) else str(src)
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        if isinstance(gt, dict):
            gt = gt.get("target")
        if isinstance(gt, str):
            gt = [gt]
        answers = [normalize(str(a)) for a in (list(gt) if gt is not None else [])]
        q = str((info or {}).get("question", "")).strip()
        gold[(ds, q)] = None if (ds, q) in gold else [a for a in answers if a]
    return {k: v for k, v in gold.items() if v}


def load(dump_dir: Path, gold: Dict[tuple, List[str]]) -> Dict[tuple, dict]:
    rows: Dict[tuple, Optional[dict]] = {}
    with resolve_dump(dump_dir).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src = str(r.get("data_source", ""))
            ds = src[len(SOURCE_PREFIX):] if src.startswith(SOURCE_PREFIX) else src
            out = str(r.get("output", ""))
            key = (ds, question_of(str(r.get("input", ""))))
            if key in rows:              # repeated question text: cannot be paired, leave it out
                rows[key] = None
                continue
            calls = search_calls(out)
            answers = gold.get(key)
            docs = retrieved_texts(out) if answers else []
            rows[key] = {
                "ds": ds, "correct": float(r["score"]) == 1.0, "searches": len(calls), "queries": sum(calls),
                "answered": bool(ANSWER.search(out)), "chars": len(out),
                "retrieved": (any(a in d for d in docs for a in answers) if answers else None)}
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
    return min(1.0, 2 * math.exp(m) * sum(math.exp(x - m) for x in logs))


def _share(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else None


def describe(rows: List[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    calls = sum(r["searches"] for r in rows)
    g = [r for r in rows if r["retrieved"] is not None]
    return {"n": n, "em": sum(r["correct"] for r in rows) / n,
            "searches_mean": calls / n, "queries_mean": sum(r["queries"] for r in rows) / n,
            "queries_per_call": (sum(r["queries"] for r in rows) / calls) if calls else None,
            "no_search_share": _share(rows, lambda r: r["searches"] == 0),
            "no_answer_share": _share(rows, lambda r: not r["answered"]),
            "chars_mean": sum(r["chars"] for r in rows) / n,
            "gold_known": len(g),
            "gold_retrieved_share": _share(g, lambda r: r["retrieved"]),
            "wrong_gold_never_retrieved": _share(g, lambda r: not r["correct"] and not r["retrieved"]),
            "wrong_gold_retrieved": _share(g, lambda r: not r["correct"] and r["retrieved"])}


BUCKETS = (("0", lambda s: s == 0), ("1", lambda s: s == 1), ("2", lambda s: s == 2), ("3+", lambda s: s >= 3))


def by_searches(rows: List[dict]) -> dict:
    out = {}
    for label, test in BUCKETS:
        sel = [r for r in rows if test(r["searches"])]
        g = [r for r in sel if r["retrieved"] is not None]
        out[label] = {"share": len(sel) / len(rows) if rows else None,
                      "em": (sum(r["correct"] for r in sel) / len(sel)) if sel else None,
                      "gold_retrieved_share": _share(g, lambda r: r["retrieved"])}
    return out


def paired(a: Dict[tuple, dict], b: Dict[tuple, dict], names) -> dict:
    keys = [k for k in a if k in b and k[0] in names]
    cw = [k for k in keys if a[k]["correct"] and not b[k]["correct"]]
    wc = [k for k in keys if not a[k]["correct"] and b[k]["correct"]]
    def split(ks):
        d = [b[k]["searches"] - a[k]["searches"] for k in ks]
        return {"fewer": sum(x < 0 for x in d), "same": sum(x == 0 for x in d), "more": sum(x > 0 for x in d)}
    return {"pairs": len(keys), "correct_to_wrong": len(cw), "wrong_to_correct": len(wc),
            "em_change": (len(wc) - len(cw)) / len(keys) if keys else None,
            "mcnemar_p": mcnemar_p(len(cw), len(wc)),
            "searches_change_mean": (sum(b[k]["searches"] - a[k]["searches"] for k in keys) / len(keys)
                                     if keys else None),
            "correct_to_wrong_by_search_change": split(cw), "wrong_to_correct_by_search_change": split(wc),
            "all_pairs_by_search_change": split(keys)}


GROUPS = {**{d: (d,) for d in DATASETS}, "single_hop": SINGLE, "multi_hop": MULTI}


def summarize_run(rows: Dict[tuple, dict]) -> dict:
    vals = list(rows.values())
    return {g: {**describe([r for r in vals if r["ds"] in names]),
                **({"em_by_searches": by_searches([r for r in vals if r["ds"] in names])}
                   if g in ("single_hop", "multi_hop") else {})}
            for g, names in GROUPS.items()}


def diagnose(eval_root: Path, test_parquet: Optional[str] = None, compare: Optional[Dict[str, Path]] = None) -> dict:
    gold = load_gold(test_parquet)
    runs: Dict[str, Dict[tuple, dict]] = {}
    for t in (0, 1, 2, 3):
        d = eval_root / f"iter{t}"
        if d.is_dir():
            try:
                runs[str(t)] = load(d, gold)
            except (OSError, ValueError) as e:
                print(f"[p0_diag] iter{t}: skipped ({e})", file=sys.stderr)
    for name, d in (compare or {}).items():
        try:
            runs[name] = load(Path(d), gold)
        except (OSError, ValueError) as e:
            print(f"[p0_diag] {name}: skipped ({e})", file=sys.stderr)
    res: dict = {"exploratory": True, "runs": list(runs), "gold_answers_loaded": len(gold),
                 "per_run": {k: summarize_run(v) for k, v in runs.items()}, "paired": {}}
    pairs = [("0", "1"), ("1", "2"), ("2", "3"), ("1", "3")]
    pairs += [(base, name) for name in (compare or {}) for base in ("2", "3")]
    for a, b in pairs:
        if a in runs and b in runs:
            res["paired"][f"{a}->{b}"] = {g: paired(runs[a], runs[b], names) for g, names in GROUPS.items()}
    return res


def _f(x, w=6, p=3):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def table(res: dict) -> str:
    lines = ["group             run      n     EM  calls/q  queries/q  q/call  no-ans  goldRet  wrongNoGold  chars"]
    for g in (*DATASETS, "single_hop", "multi_hop"):
        for k in res["runs"]:
            s = res["per_run"][k].get(g, {})
            if s.get("n"):
                lines.append(f"{g:17s} {k:>6s} {s['n']:6d} {_f(s['em'])} {_f(s['searches_mean'], 8, 2)}"
                             f" {_f(s['queries_mean'], 10, 2)} {_f(s['queries_per_call'], 7, 2)}"
                             f" {_f(s['no_answer_share'], 7)} {_f(s['gold_retrieved_share'], 8)}"
                             f" {_f(s['wrong_gold_never_retrieved'], 12)} {s['chars_mean']:6.0f}")
    for k, v in res["paired"].items():
        for g in ("single_hop", "multi_hop"):
            p = v[g]
            if p["pairs"]:
                cw, wc = p["correct_to_wrong_by_search_change"], p["wrong_to_correct_by_search_change"]
                lines.append(f"paired {k:>9s} {g:10s}: pairs {p['pairs']}, c->w {p['correct_to_wrong']} "
                             f"(fewer/same/more searches {cw['fewer']}/{cw['same']}/{cw['more']}), "
                             f"w->c {p['wrong_to_correct']} ({wc['fewer']}/{wc['same']}/{wc['more']}), "
                             f"McNemar p {'-' if p['mcnemar_p'] is None else format(p['mcnemar_p'], '.2g')}, "
                             f"searches {p['searches_change_mean']:+.3f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--test-parquet")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    compare = dict(c.split("=", 1) for c in a.compare)
    res = diagnose(Path(a.eval_root), a.test_parquet, {k: Path(v) for k, v in compare.items()})
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(table(res))
    print(f"\nwritten: {a.out}  (please send this JSON file itself, not a retyped table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
