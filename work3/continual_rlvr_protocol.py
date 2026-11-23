"""Frozen pilot settings and decisions. Imported by the CPU and CUDA paths."""

import math
import random
from statistics import mean

PROTOCOL = "continual_rlvr_v1"
SEEDS = (1729, 2833)
ARMS = ("sequential", "static_mix", "prompt_replay", "trace_replay", "uniform_prefix", "guard")
A_STEPS = 48
B_STEPS = 32
GROUP = 8
PREFIX = 8
CAP = {"a": 384, "b": 256}
BASE_EVAL = {"a": 4, "b": 16}
METHOD_EVAL = 4
PRETRAIN_BANK = 16
PREFIX_SAMPLES = 8
MAX_PREFIXES = 2
BATCH_EVAL = 16
LR = 2e-5
GUARD_FLOOR = 0.75
BUDGET = {"necessity": 48 * 3600, "method": 12 * 3600}


def check_scores(rows, cases, samples):
    expected = {case["id"] for case in cases}
    if set(rows) != expected or any(len(rewards) != samples or
                                    any(type(reward) is not bool for reward in rewards)
                                    for rewards in rows.values()):
        raise ValueError("missing, unexpected or malformed evaluation rollouts")


def pass_one(rows):
    return mean(mean(rewards) for rewards in rows.values())


def coverage(rows, start=0, stop=None):
    return mean(any(rewards[start:stop]) for rewards in rows.values())


def paired_interval(deltas):
    rng = random.Random(19021)
    means = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    return [means[50], means[1949]]


def necessity_verdict(before, after):
    a_gain = pass_one(after["a"]) - pass_one(before["a"])
    losses, base = [], []
    for start in (0, 16):
        base.append(coverage(before["b"], start, start + 16))
        losses.append(base[-1] - coverage(after["b"], start, start + 16))
    deltas = [mean(int(any(before["b"][key][start:start + 16])) -
                   int(any(after["b"][key][start:start + 16])) for start in (0, 16))
              for key in before["b"]]
    checks = {"a_gain": a_gain >= 0.05 - 1e-12, "b_base": min(base) >= 0.20,
              "b_drop": mean(losses) >= 0.05 - 1e-12 and min(losses) > 0}
    return {"verdict": "GO" if all(checks.values()) else "NO_GO", "checks": checks,
            "a_pass1_gain": a_gain, "b_base_pass16": base, "b_pass16_drops": losses,
            "b_paired_interval": paired_interval(deltas)}


def rank_correlation(pairs):
    def ranks(values):
        result = [0.0] * len(values)
        ordered = sorted(range(len(values)), key=values.__getitem__)
        start = 0
        while start < len(ordered):
            stop = start + 1
            while stop < len(ordered) and values[ordered[stop]] == values[ordered[start]]:
                stop += 1
            for index in ordered[start:stop]:
                result[index] = (start + stop - 1) / 2
            start = stop
        return result

    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    xs, ys = ranks(xs), ranks(ys)
    mid = (len(pairs) - 1) / 2
    denominator = math.sqrt(sum((x - mid) ** 2 for x in xs) * sum((y - mid) ** 2 for y in ys))
    return sum((x - mid) * (y - mid) for x, y in zip(xs, ys)) / denominator if denominator else None


def method_verdict(records):
    if set(records) != set(SEEDS) or any(set(arms) != set(ARMS) for arms in records.values()):
        raise ValueError("all frozen seeds and arms are required")
    references = ARMS[:-1]
    best = max(references, key=lambda arm: mean(records[seed][arm]["b_pass1"] for seed in SEEDS))
    gains = [records[seed]["guard"]["b_pass1"] - records[seed][best]["b_pass1"] for seed in SEEDS]
    a_losses = [records[seed]["sequential"]["a_after_a_pass1"] -
                records[seed]["guard"]["a_after_a_pass1"] for seed in SEEDS]
    learned = {arm: [records[seed][arm]["b_pass1"] - records[seed][arm]["b_curve"][0]
                     for seed in SEEDS] for arm in ("guard", best)}
    deltas = [mean(mean(records[seed]["guard"]["b_final"][key]) -
                   mean(records[seed][best]["b_final"][key]) for seed in SEEDS)
              for key in records[SEEDS[0]]["guard"]["b_final"]]
    # Each arm receives the same number of update, generation, and auxiliary
    # forward slots. Extra actual resource use must also stay within tolerance.
    cost_keys = ("a_rollout_tokens", "a_update_tokens", "b_rollout_tokens", "b_update_tokens", "train_seconds")
    ratios = {str(seed): {key: records[seed]["guard"]["cost"][key] /
                          max(1, records[seed][best]["cost"][key])
                          for key in cost_keys} for seed in SEEDS}
    correlations = [rank_correlation([(arms[arm]["calibration_ratio"], arms[arm]["b_after_a_pass4"])
                                      for arm in ARMS]) for arms in records.values()]
    checks = {"b_gain": mean(gains) >= 0.03 - 1e-12 and min(gains) > 0,
              "b_learning": mean(learned["guard"]) > 0 and mean(learned["guard"]) >= mean(learned[best]) - 1e-12,
              "a_retention": max(a_losses) <= 0.015 + 1e-12,
              "cost": all(max(seed.values()) <= 1.10 for seed in ratios.values()),
              "proxy": all(value is not None and value >= 0.30 for value in correlations)}
    return {"verdict": "GO" if all(checks.values()) else "NO_GO", "checks": checks,
            "best_reference": best, "b_pass1_gains": gains, "b_paired_interval": paired_interval(deltas),
            "b_gain_from_own_start": learned,
            "a_pass1_losses": a_losses, "cost_ratios": ratios, "proxy_correlations": correlations}
