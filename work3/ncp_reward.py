"""Proposer rewards for the P2 arms (protocol research/protocols/P2_ncp_pilot.md).

Drop-in replacement for Dr. Zero's `compute_challenger_score_batch`
(verl/custom_reward/reward_function.py): same format score, same question /
answer extraction, same final combination `0.5 * format + difficulty`; only the
difficulty term changes per arm:

  drzero  n=5  I(0<k<n) (n-k)/(n-1)                       (upstream, for reference)
  beta    n=7  I(0<k<n) (1-p~)/(1-p~_min),  p~ = (k+1)/(n+2)  (Laplace / Beta(1,1))
  ncp     n=5  drzero difficulty x w,  w = max(0, s2_tot - s2_noise) / s2_tot
               s2_tot   = k(n-k)/(n(n-1))
               s2_noise = (1/n) sum_i (Rt_i1 - Rt_i2)^2 / 2  (2 same-prefix replays per
               solver rollout with permuted passages, exactly as in P1)

Solver rollouts use work3.p1_agent (prompt construction verified identical to
upstream). Selected via the launch-script overrides
  custom_reward_function.name=compute_challenger_score_batch_arm
  custom_reward_function.path=<work3>/work3/ncp_reward.py
  +custom_reward_function.reward_kwargs.arm=<arm>
Per-call diagnostics go to $WORK3_REWARD_LOG (JSONL) when set.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:          # loaded by file path inside a Ray worker
    sys.path.insert(0, _ROOT)

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from work3 import p1_agent as A  # noqa: E402

ARMS = {"drzero": 5, "beta": 7, "ncp": 5}
N_REPLAY = 2
QUESTION_CONCURRENCY = 256
SEED = 20261106
_CALLS = {"n": 0}


# ---------------------------------------------------------------- difficulty terms
def drzero_difficulty(k: int, n: int) -> float:
    return 0.0 if k <= 0 or k >= n else (n - k) / (n - 1)


def beta_difficulty(k: int, n: int) -> float:
    if k <= 0 or k >= n:
        return 0.0
    p, p_min = (k + 1) / (n + 2), 2 / (n + 2)
    return (1 - p) / (1 - p_min)


def signal_share(scores: list[int], pairs: list[tuple[int, int]]) -> tuple[float, float, float]:
    """(w, s2_tot, s2_noise) for one question."""
    n, k = len(scores), int(sum(scores))
    tot = k * (n - k) / (n * (n - 1))
    noise = float(np.mean([(a - b) ** 2 / 2 for a, b in pairs])) if pairs else 0.0
    if tot == 0:
        return 0.0, tot, noise
    return max(0.0, tot - noise) / tot, tot, noise


def arm_reward(arm: str, scores: list[int], pairs: list[tuple[int, int]] | None = None) -> dict:
    n, k = len(scores), int(sum(scores))
    out = {"k": k, "n": n}
    if arm == "drzero":
        out["difficulty"] = drzero_difficulty(k, n)
    elif arm == "beta":
        out["difficulty"] = beta_difficulty(k, n)
    elif arm == "ncp":
        w, tot, noise = signal_share(scores, pairs or [])
        out.update({"w": w, "s2_tot": tot, "s2_noise": noise,
                    "difficulty": drzero_difficulty(k, n) * w})
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return out


# ---------------------------------------------------------------- rollouts
async def score_question(arm: str, messages: list[dict], answer: str, renderer: A.Renderer,
                         clients, seed: list[int]) -> dict:
    n = ARMS[arm]
    eps = await asyncio.gather(*[A.run_episode(messages, renderer, clients.generate, clients.search)
                                 for _ in range(n)])
    scores = [ep.score(answer) for ep in eps]
    pairs, stats = None, {"unreplayable": 0, "new_search": 0}
    if arm == "ncp":
        async def one(i: int, j: int) -> int:
            prep = A.replay_prefix(eps[i], np.random.default_rng([*seed, i, j]))
            if prep is None:
                stats["unreplayable"] += 1
                return scores[i]
            prefix, results, used, _ = prep
            ep = await A.run_episode(prefix, renderer, clients.generate, clients.search,
                                     turns_used=used, tool_results=results)
            stats["new_search"] += int(ep.new_tool_calls > 0)
            return ep.score(answer)
        reps = await asyncio.gather(*[asyncio.gather(*[one(i, j) for j in range(N_REPLAY)])
                                      for i in range(n)])
        pairs = [tuple(r) for r in reps]
    out = arm_reward(arm, scores, pairs)
    out.update(stats)
    return out


async def score_all(arm: str, questions: list[str], answers: list[str], renderer: A.Renderer,
                    make_clients, call_idx: int, solver_prefix: str) -> tuple[list[dict], dict]:
    clients = make_clients()
    sem = asyncio.Semaphore(QUESTION_CONCURRENCY)

    async def one(i: int, q: str, a: str) -> dict:
        async with sem:
            msgs = [A.message("user", solver_prefix.format(question=q.strip()))]
            return await score_question(arm, msgs, a, renderer, clients, [SEED, call_idx, i])
    try:
        res = await asyncio.gather(*[one(i, q, a) for i, (q, a) in enumerate(zip(questions, answers))])
    finally:
        close = getattr(getattr(clients, "client", None), "aclose", None)
        if close:
            await close()
    return res, dict(getattr(clients, "calls", {}))


def _search_url(tool_config_path: str) -> str:
    import yaml

    with open(tool_config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["tools"][0]["config"]["retrieval_service_url"]


def _log(record: dict) -> None:
    path = os.environ.get("WORK3_REWARD_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def compute_challenger_score_batch_arm(data_sources, solution_strs, ground_truths, extra_infos,
                                       **kwargs):
    from verl.custom_reward.reward_function import compute_challenger_format_scores
    from verl.prompts import DEFAULT_SOLVER_PREFIX

    from work3.p1_run import Clients

    t0 = time.time()
    arm = str(kwargs.get("arm", "ncp"))
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    batch, tok, config = kwargs["data"], kwargs["processing_class"], kwargs["config"]
    hops = [int(x.split("_")[-1]) for x in batch.non_tensor_batch["data_source"]]
    raw_messages = [tok.decode(batch.batch["input_ids"][i]) for i in range(len(batch))]
    responses = [tok.decode(batch.batch["responses"][i], skip_special_tokens=True)
                 for i in range(len(batch))]
    format_scores, raw_qs, raw_ans = compute_challenger_format_scores(
        raw_messages, responses, hops, return_qa=True)
    valid = [i for i, s in enumerate(format_scores) if s > 0]
    if not valid:
        return format_scores                     # upstream returns unscaled scores here

    tool_cfg = config["multi_turn"]["tool_config_path"]
    tools, schema_source = A.upstream_tool_schemas(tool_cfg)
    renderer = A.Renderer(tok, tools)
    search_url = _search_url(tool_cfg)
    _CALLS["n"] += 1
    call_idx = _CALLS["n"]

    def make_clients():
        return Clients(kwargs["base_url"], kwargs["model_name"], search_url, QUESTION_CONCURRENCY)

    try:                       # upstream pattern; uvloop >= 0.22 no longer creates a loop here
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    results, calls = loop.run_until_complete(score_all(
        arm, [raw_qs[i] for i in valid], [raw_ans[i] for i in valid], renderer, make_clients,
        call_idx, DEFAULT_SOLVER_PREFIX))

    final = [0.5 * x for x in format_scores]
    for i, r in zip(valid, results):
        final[i] += r["difficulty"]
    k_hist = {str(k): sum(r["k"] == k for r in results) for k in range(ARMS[arm] + 1)}
    rec = {"arm": arm, "call": call_idx, "batch": len(format_scores), "valid": len(valid),
           "k_hist": k_hist, "mean_difficulty": float(np.mean([r["difficulty"] for r in results])),
           "mean_final": float(np.mean(final)), "calls": calls, "schema": schema_source,
           "seconds": round(time.time() - t0, 1)}
    if arm == "ncp":
        mixed = [r for r in results if 0 < r["k"] < r["n"]]
        rec.update({
            "mean_w_mixed": float(np.mean([r["w"] for r in mixed])) if mixed else None,
            "eta_mixed": (sum(r["s2_noise"] for r in mixed) / sum(r["s2_tot"] for r in mixed)
                          if mixed else None),
            "unreplayable": sum(r["unreplayable"] for r in results),
            "new_search_replays": sum(r["new_search"] for r in results),
        })
    _log(rec)
    print(f"[work3 reward {arm}] call {call_idx}: valid {len(valid)}/{len(format_scores)} "
          f"k {k_hist} mean difficulty {rec['mean_difficulty']:.3f} ({rec['seconds']} s)", flush=True)
    return final
