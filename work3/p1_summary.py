"""P1 summary and gate (protocol research/protocols/P1_noise_diagnosis.md, frozen).

    python -m work3.p1_summary --run-dir OUT [--pool-check P0_OUT/pool_check_all.json]
        [--parity OUT/p1_parity.json] --out OUT/p1_summary.json

Per question (n = 5 originals, R_i in {0,1}, k = sum R_i, 2 replays per original):
  sigma2_tot   = k (n - k) / (n (n - 1))
  sigma2_noise = (1/n) sum_i (Rt_i1 - Rt_i2)^2 / 2
Group eta(G) = sum sigma2_noise / sum sigma2_tot over questions in G with 0 < k < n.
Primary (pools 1 and 3 pooled): eta(k=1) >= 0.35 and eta(k=1) - eta(k in {3,4}) >= 0.10
with the 95% bootstrap interval of the difference above 0 (question-level, 2000
resamples within each pool, numpy seed 20261106). Unreplayable originals keep their
own outcome for both replays (zero noise contribution; reported).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N = 5
ETA_MIN = 0.35
DIFF_MIN = 0.10
NEW_SEARCH_MAX = 0.20
MIN_GROUP = 60
MAX_QUESTIONS = 1500
BOOT = 2000
SEED = 20261106
POOLS = (1, 3)


def question_stats(rec: dict) -> dict:
    scores = [int(s) for s in rec["scores"]]
    assert len(scores) == N, rec["qid"]
    k = sum(scores)
    tot = k * (N - k) / (N * (N - 1))
    noise_terms, new_search, replays, unreplayable = [], 0, 0, 0
    for i, pair in enumerate(rec["replays"]):
        assert len(pair) == 2, rec["qid"]
        a, b = int(pair[0]["score"]), int(pair[1]["score"])
        noise_terms.append((a - b) ** 2 / 2)
        for r in pair:
            replays += 1
            new_search += int(r.get("new_tool_calls", 0) > 0)
            unreplayable += int(not r.get("replayable", True))
    return {"qid": rec["qid"], "k": k, "tot": tot, "noise": float(np.mean(noise_terms)),
            "replays": replays, "new_search": new_search, "unreplayable": unreplayable}


def eta(stats: list[dict], ks: tuple[int, ...]) -> float | None:
    sel = [s for s in stats if s["k"] in ks]
    tot = sum(s["tot"] for s in sel)
    return None if tot == 0 else sum(s["noise"] for s in sel) / tot


def bootstrap(pools: dict[int, list[dict]], boot: int = BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    arrays = {}
    for p, st in pools.items():
        k = np.array([s["k"] for s in st]); arrays[p] = (
            k, np.array([s["tot"] for s in st]), np.array([s["noise"] for s in st]))
    e1s, diffs = [], []
    for _ in range(boot):
        num1 = den1 = num34 = den34 = 0.0
        for p, (k, tot, noise) in arrays.items():
            idx = rng.integers(0, len(k), size=len(k))
            kk, tt, nn = k[idx], tot[idx], noise[idx]
            m1, m34 = kk == 1, (kk == 3) | (kk == 4)
            num1 += nn[m1].sum(); den1 += tt[m1].sum()
            num34 += nn[m34].sum(); den34 += tt[m34].sum()
        if den1 == 0 or den34 == 0:
            continue
        e1, e34 = num1 / den1, num34 / den34
        e1s.append(e1); diffs.append(e1 - e34)
    pct = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None
    return {"resamples_used": len(diffs), "eta_k1_ci95": pct(e1s), "diff_ci95": pct(diffs)}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(run_dir: Path, pool_check_ok: bool | None, parity: dict | None) -> dict:
    problems, per_pool, pools = [], {}, {}
    all_replays = all_new = all_unrep = 0
    lucky_all = []
    for p in POOLS:
        recs = {r["qid"]: r for r in read_jsonl(run_dir / f"pool{p}_questions.jsonl")}
        state_path = run_dir / f"pool{p}_state.json"
        if not recs or not state_path.exists():
            problems.append(f"pool{p}: rollouts or state missing")
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        sampled = [q for b in state["batches"] for q in b["qids"]]
        missing = [q for q in sampled if q not in recs]
        if missing:
            problems.append(f"pool{p}: {len(missing)} sampled questions without rollouts")
        st = [question_stats(recs[q]) for q in sampled if q in recs]
        pools[p] = st
        n1 = sum(s["k"] == 1 for s in st); n34 = sum(s["k"] in (3, 4) for s in st)
        if (n1 < MIN_GROUP or n34 < MIN_GROUP):
            problems.append(f"pool{p}: group sizes k=1 {n1}, k in 3-4 {n34} below {MIN_GROUP} "
                            f"after {len(sampled)} questions")
        lucky = {r["qid"]: r for r in read_jsonl(run_dir / f"pool{p}_lucky.jsonl")}
        k1_qids = [s["qid"] for s in st if s["k"] == 1]
        if any(q not in lucky for q in k1_qids):
            problems.append(f"pool{p}: lucky-hit rollouts incomplete")
        lucky_succ = [sum(lucky[q]["scores"]) for q in k1_qids if q in lucky]
        lucky_all += lucky_succ
        reps = sum(s["replays"] for s in st); new = sum(s["new_search"] for s in st)
        unrep = sum(s["unreplayable"] for s in st)
        all_replays += reps; all_new += new; all_unrep += unrep
        finishes: dict[str, int] = {}
        for q in sampled:
            for ep in recs.get(q, {}).get("originals", []):
                finishes[ep["finish"]] = finishes.get(ep["finish"], 0) + 1
        per_pool[str(p)] = {
            "questions": len(st), "batches": len(state["batches"]),
            "k_hist": {str(k): sum(s["k"] == k for s in st) for k in range(N + 1)},
            "eta_k1": eta(st, (1,)), "eta_k34": eta(st, (3, 4)), "eta_mixed": eta(st, (1, 2, 3, 4)),
            "replays": reps, "new_search_replays": new, "unreplayable_replays": unrep,
            "lucky_share_le_1_of_16": (float(np.mean([s <= 1 for s in lucky_succ]))
                                       if lucky_succ else None),
            "original_finish": finishes, "tool_schema_source": state.get("tool_schema_source"),
            "calls": state.get("calls"),
        }
    if pool_check_ok is None:
        problems.append("P0 pool check not supplied")
    elif not pool_check_ok:
        problems.append("P0 pool check failed")
    new_rate = all_new / all_replays if all_replays else None
    if new_rate is not None and new_rate > NEW_SEARCH_MAX:
        problems.append(f"replays issuing a new search {new_rate:.3f} > {NEW_SEARCH_MAX}")

    result = {"protocol": "research/protocols/P1_noise_diagnosis.md", "pools": per_pool,
              "replay_new_search_rate": new_rate, "unreplayable_replays": all_unrep,
              "parity": ({k: parity.get(k) for k in ("checked", "text_equal", "ids_equal", "errors",
                                                      "tool_schema_source")} if parity else None)}
    if len(pools) == len(POOLS):
        pooled = [s for p in POOLS for s in pools[p]]
        e1, e34 = eta(pooled, (1,)), eta(pooled, (3, 4))
        boot = bootstrap(pools)
        diff = None if e1 is None or e34 is None else e1 - e34
        primary_a = e1 is not None and e1 >= ETA_MIN
        primary_b = (diff is not None and diff >= DIFF_MIN and boot["diff_ci95"] is not None
                     and boot["diff_ci95"][0] > 0)
        m1, m3 = per_pool["1"]["eta_mixed"], per_pool["3"]["eta_mixed"]
        result.update({
            "pooled": {"eta_k1": e1, "eta_k34": e34, "diff": diff, **boot},
            "primary": {"a_eta_k1_ge_0.35": primary_a, "b_diff_ge_0.10_ci_above_0": primary_b},
            "secondary": {
                "c_pool3_mixed_ge_pool1": (None if m1 is None or m3 is None else m3 >= m1),
                "d_lucky_share": (float(np.mean([s <= 1 for s in lucky_all])) if lucky_all else None),
                "d_lucky_share_ge_0.30": (bool(np.mean([s <= 1 for s in lucky_all]) >= 0.30)
                                          if lucky_all else None),
            },
        })
        verdict = "INVALID" if problems else ("GO" if primary_a and primary_b else "NO-GO")
    else:
        verdict = "INVALID"
    result["problems"] = problems
    result["verdict"] = verdict
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pool-check")
    ap.add_argument("--parity")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    pool_ok = None
    if args.pool_check:
        pool_ok = bool(json.loads(Path(args.pool_check).read_text(encoding="utf-8"))["ok"])
    parity = json.loads(Path(args.parity).read_text(encoding="utf-8")) if args.parity else None
    res = summarize(Path(args.run_dir), pool_ok, parity)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if res["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
