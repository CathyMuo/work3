"""P2 gate: noise-calibrated proposer reward vs Dr. Zero and Beta smoothing.

Decision rule frozen in research/protocols/P2_ncp_pilot.md.

    python -m work3.p2_gate --drzero P0/p0_summary.json --beta BETA/p0_summary.json \
        --ncp NCP/p0_summary.json [--drzero-p1 ..] [--beta-p1 ..] [--ncp-p1 ..] \
        [--beta-reward-log ..] [--ncp-reward-log ..] --out p2_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MARGIN_DRZERO = 0.020
MARGIN_BETA = 0.010
ETA_DROP = 0.10
EPS = 1e-12


def averages(summary: dict) -> dict[int, float | None]:
    its = summary.get("iterations", {})
    return {t: (its.get(str(t)) or {}).get("average") for t in (1, 2, 3)}


def eta_pool3(p1: dict | None) -> float | None:
    if not p1:
        return None
    return ((p1.get("pools") or {}).get("3") or {}).get("eta_mixed")


def reward_usage(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    calls = gen = comp = 0
    secs = 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            calls += 1
            gen += int((r.get("calls") or {}).get("generate", 0))
            comp += int((r.get("calls") or {}).get("completion_tokens", 0))
            secs += float(r.get("seconds", 0))
    return {"reward_calls": calls, "generate_requests": gen, "completion_tokens": comp,
            "reward_seconds": round(secs, 1)}


def decide(dz: dict, beta: dict, ncp: dict, p1: dict | None = None, logs: dict | None = None) -> dict:
    problems = []
    arms = {"drzero": dz, "beta": beta, "ncp": ncp}
    avgs = {}
    if dz.get("verdict") != "PASS":
        problems.append(f"drzero: P0 verdict is {dz.get('verdict')}, P2 needs a passing reproduction")
    for name, s in arms.items():
        if s.get("verdict") == "INVALID":
            problems.append(f"{name}: P0-style summary is INVALID")
        avgs[name] = averages(s)
        for t in (2, 3):
            if avgs[name][t] is None:
                problems.append(f"{name}: iteration-{t} average missing")
    res = {"protocol": "research/protocols/P2_ncp_pilot.md", "averages": avgs, "problems": problems}
    if problems:
        res["verdict"] = "INVALID"
        return res
    a3 = {k: v[3] for k, v in avgs.items()}
    d_dz, d_beta = a3["ncp"] - a3["drzero"], a3["ncp"] - a3["beta"]
    no_decline = avgs["ncp"][3] + EPS >= avgs["ncp"][2]
    prim = {"ncp_minus_drzero": d_dz, "ncp_minus_beta": d_beta,
            "a_ge_drzero_plus_0.020": d_dz + EPS >= MARGIN_DRZERO,
            "b_ge_beta_plus_0.010": d_beta + EPS >= MARGIN_BETA,
            "c_no_decline_iter2_to_3": no_decline}
    per_dataset = {}
    for d in (dz.get("iterations", {}).get("3", {}).get("em") or {}):
        per_dataset[d] = {name: arms[name]["iterations"]["3"]["em"].get(d) for name in arms}
    etas = {k: eta_pool3((p1 or {}).get(k)) for k in arms}
    logs = logs or {}
    tb = (logs.get("beta") or {}).get("completion_tokens")
    tn = (logs.get("ncp") or {}).get("completion_tokens")
    ratio = tn / tb if tb and tn is not None else None
    sec = {"eta_pool3_mixed": etas,
           "ncp_eta_drop_ge_0.10": (None if etas["drzero"] is None or etas["ncp"] is None
                                    else etas["drzero"] - etas["ncp"] >= ETA_DROP),
           "reward_usage": logs,
           "ncp_over_beta_completion_tokens": ratio,
           "compute_matched": None if ratio is None else ratio <= 1.5}
    res.update({"primary": prim, "per_dataset_iter3": per_dataset, "secondary": sec,
                "verdict": "GO" if all(prim[k] for k in prim if k[:2] in ("a_", "b_", "c_")) else "NO-GO"})
    return res


def _load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("drzero", "beta", "ncp"):
        ap.add_argument(f"--{a}", required=True)
        ap.add_argument(f"--{a}-p1")
    for a in ("beta", "ncp"):
        ap.add_argument(f"--{a}-reward-log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    summaries = {}
    for a in ("drzero", "beta", "ncp"):
        s = _load(getattr(args, a))
        summaries[a] = s if s is not None else {"verdict": "INVALID"}
    p1 = {a: _load(getattr(args, f"{a}_p1")) for a in ("drzero", "beta", "ncp")}
    logs = {a: reward_usage(getattr(args, f"{a}_reward_log")) for a in ("beta", "ncp")}
    res = decide(summaries["drzero"], summaries["beta"], summaries["ncp"], p1, logs)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if res["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
