"""Frozen equal-cost comparison on a fully observed short/long rollout panel."""

import math
import random
from collections import defaultdict

PROTOCOL = "budget_rlvr_paired_pilot_v1"
QUESTIONS = 256
REPEATS = 8
PREFIX = 384
SHORT = 64
LONG = 1024
FOLDS = 4
AUDIT_SHARE = 0.50
P_MIN = 0.05
COST_TOLERANCE = 0.03
RESCUE_MIN = 0.05
GAIN_MIN = 0.20
METHOD = "paired_disagreement"
BASELINES = ("uniform", "prefix_critic", "paired_uniform", "short_calibrated_critic", "full_long")


def exact_panel(rows, cases, repeats):
    keys = [(row["qid"], row["repeat"]) for row in rows]
    expected = {(case["id"], n) for case in cases for n in range(repeats)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete or duplicate rollout panel")
    for row in rows:
        boolean_keys = ("prefix_ended", "prefix_correct", "prefix_has_box", "short_ended", "short_correct",
                        "long_ended", "long_correct")
        if any(type(row[key]) is not bool for key in boolean_keys):
            raise ValueError("malformed rollout flags")
        prefix = row["prefix_ids"]
        if (not 1 <= len(prefix) <= PREFIX or (not row["prefix_ended"] and len(prefix) != PREFIX)
                or not 1 <= row["prompt_tokens"] <= 768
                or row["base_tokens"] != row["prompt_tokens"] + len(prefix)
                or row["long_correct"] and not row["long_ended"]):
            raise ValueError("invalid prefix or costs")
        if row["prefix_ended"]:
            if (row["short_tokens"] or row["long_tokens"] or row["short_ids"] or row["long_ids"]
                    or not row["long_ended"] or not row["short_ended"]
                    or row["long_correct"] != row["prefix_correct"] or row["short_correct"] != row["prefix_correct"]):
                raise ValueError("an already ended prefix has inconsistent outcomes")
        elif (not 1 <= len(row["short_ids"]) <= SHORT or not 1 <= len(row["long_ids"]) <= LONG - PREFIX
              or (not row["short_ended"] and len(row["short_ids"]) != SHORT)
              or (not row["long_ended"] and len(row["long_ids"]) != LONG - PREFIX)
              or row["short_tokens"] <= len(row["short_ids"]) or row["long_tokens"] != len(row["long_ids"])):
            raise ValueError("missing branch outputs or charged continuation costs")


def features(row, short=False):
    ids = row["prefix_ids"]
    key = (row["prefix_correct"], row["prefix_has_box"], row["prompt_tokens"] >= 128,
           len(set(ids)) * 3 >= len(ids))
    return (*key, row["short_correct"]) if short else key


def fit(rows, outcome, short=False):
    values = [outcome(row) for row in rows]
    mean = sum(values) / len(values)
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[features(row, short)].append(value)

    def predict(row):
        cell = groups.get(features(row, short), ())
        return (sum(cell) + 8 * mean) / (len(cell) + 8)

    return predict


def propensities(costs, risk, budget):
    if budget < P_MIN * sum(costs) - 1e-8:
        return None
    if budget >= sum(costs):
        return [1.0] * len(costs)
    scales = [math.sqrt(max(1e-4, r) / c) for c, r in zip(costs, risk)]
    low, high = 0.0, 1 / min(scales)
    for _ in range(48):
        mid = (low + high) / 2
        allocated = sum(c * min(1, max(P_MIN, mid * s)) for c, s in zip(costs, scales))
        if allocated > budget:
            high = mid
        else:
            low = mid
    return [min(1, max(P_MIN, low * s)) for s in scales]


def fold_comparison(train, test):
    fit_open = [row for row in train if not row["prefix_ended"]]
    open_rows = [row for row in test if not row["prefix_ended"]]
    if not fit_open or not open_rows:
        raise ValueError("no truncated prefixes in a cross-fit fold")
    tail = fit(fit_open, lambda row: row["long_tokens"])
    hidden_reward = fit(fit_open, lambda row: row["long_correct"])
    short_reward = fit(fit_open, lambda row: row["long_correct"], short=True)
    disagreement = fit(fit_open, lambda row: row["long_correct"] != row["short_correct"], short=True)
    costs = [tail(row) for row in open_rows]
    base = sum(row["base_tokens"] for row in test)
    available = AUDIT_SHARE * sum(costs)
    target = base + available
    plans = {}
    for name in (*BASELINES, METHOD):
        if name == "full_long":
            # Draw full rollouts from scratch, including easy cases. Its cost prediction
            # comes from other questions, before drawing any held-out prefixes.
            mean_full = sum(row["base_tokens"] + row["long_tokens"] for row in train) / len(train)
            q = min(1.0, target / (len(test) * mean_full))
            plans[name] = [(row["qid"], q * (row["base_tokens"] + row["long_tokens"]),
                            (1 / q - 1) * row["long_correct"]) for row in test]
            continue
        short = name in {"paired_uniform", "short_calibrated_critic", METHOD}
        budget = available - (sum(row["short_tokens"] for row in open_rows) if short else 0)
        if budget < P_MIN * sum(costs):
            plans[name] = None
            continue
        if name in {"uniform", "paired_uniform"}:
            q = [min(1.0, budget / sum(costs))] * len(open_rows)
        else:
            critic = short_reward if short else hidden_reward
            risk = [disagreement(row) if name == METHOD else critic(row) * (1 - critic(row))
                    for row in open_rows]
            q = propensities(costs, risk, budget)
        pieces = [(row["qid"], row["base_tokens"], 0.0) for row in test if row["prefix_ended"]]
        for row, prob in zip(open_rows, q):
            proxy = 0.0
            if name in {METHOD, "paired_uniform"}:
                proxy = row["short_correct"]
            elif name in {"prefix_critic", "short_calibrated_critic"}:
                proxy = (short_reward if short else hidden_reward)(row)
            pieces.append((row["qid"], row["base_tokens"] + (row["short_tokens"] if short else 0)
                           + prob * row["long_tokens"], (1 / prob - 1) * (row["long_correct"] - proxy) ** 2))
        plans[name] = pieces
    return target, plans


def compare(rows, cases):
    exact_panel(rows, cases, REPEATS)
    folds = {case["id"]: index % FOLDS for index, case in enumerate(cases)}
    by_fold = [[row for row in rows if folds[row["qid"]] == index] for index in range(FOLDS)]
    planned, infeasible, target = defaultdict(list), set(), 0.0
    for index, test in enumerate(by_fold):
        train = [row for j, bucket in enumerate(by_fold) if index != j for row in bucket]
        if not any(not row["prefix_ended"] for row in test):
            return {"verdict": "NO_GO", "reason": "no_truncated_prefixes_in_fold", **observed(rows)}
        fold_target, result = fold_comparison(train, test)
        target += fold_target
        for name, pieces in result.items():
            if pieces is None:
                infeasible.add(name)
            else:
                planned[name].extend(pieces)
    sample_count = len(rows)
    arms = {}
    variances = {}
    for name, pieces in planned.items():
        if name in infeasible:
            continue
        by_question = defaultdict(float)
        for qid, _, variance in pieces:
            by_question[qid] += variance
        variances[name] = by_question
        arms[name] = {"expected_tokens": sum(cost for _, cost, _ in pieces),
                      "rmse": math.sqrt(sum(by_question.values())) / sample_count}
    counts = observed(rows)
    available = [arm for arm in BASELINES if arm in arms]
    best = min(available, key=lambda name: arms[name]["rmse"]) if available else None
    # An infeasible optional proxy comparator cannot create a positive claim.
    cost_match = (len(arms) == len(BASELINES) + 1
                  and max(arm["expected_tokens"] for arm in arms.values())
                  <= (1 + COST_TOLERANCE) * min(arm["expected_tokens"] for arm in arms.values()))
    gain, interval = None, None
    if METHOD in arms and best is not None and arms[best]["rmse"] > 0:
        gain = 1 - arms[METHOD]["rmse"] / arms[best]["rmse"]
        rng = random.Random(14119)
        ids = list(folds)
        values = []
        for _ in range(2000):
            panel = rng.choices(ids, k=len(ids))
            ref = sum(variances[best][qid] for qid in panel)
            if ref > 0:
                values.append(1 - math.sqrt(sum(variances[METHOD][qid] for qid in panel) / ref))
        if values:
            values.sort()
            interval = [values[int(0.025 * len(values))], values[int(0.975 * len(values))]]
    checks = {"late_rescue": counts["late_rescue_rate"] >= RESCUE_MIN, "matched_cost": cost_match,
              "rmse_gain": gain is not None and gain >= GAIN_MIN}
    disposition = "GO" if all(checks.values()) else "NO_GO"
    return {**counts, "verdict": disposition, "checks": checks, "planned_expected_tokens": target,
            "infeasible": sorted(infeasible), "arms": arms, "best_baseline": best,
            "rmse_relative_gain": gain, "question_bootstrap_interval": interval}


def observed(rows):
    unfinished = [row for row in rows if not row["prefix_ended"]]
    late = sum(not row["prefix_correct"] and not row["short_correct"] and row["long_correct"] for row in unfinished)
    return {"samples": len(rows), "truncated_at_prefix": len(unfinished),
            "correct_prefix_without_eos": sum(row["prefix_correct"] for row in unfinished),
            "short_wrong_long_right": sum(not row["short_correct"] and row["long_correct"] for row in unfinished),
            "short_right_long_wrong": sum(row["short_correct"] and not row["long_correct"] for row in unfinished),
            "short_ended": sum(row["short_ended"] for row in unfinished),
            "long_ended": sum(row["long_ended"] for row in unfinished),
            "long_reward_mean": sum(row["long_correct"] for row in rows) / len(rows),
            "late_rescue_count": late, "late_rescue_rate": late / len(rows),
            "actual_panel_tokens": sum(row["base_tokens"] + row["short_tokens"] + row["long_tokens"] for row in rows)}
