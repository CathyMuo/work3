"""Crossed feedback features, equal-cap baselines and sealed paired inference."""

import hashlib
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from statistics import fmean
from time import monotonic

import numpy as np


PAIRS = ("f1p1", "f1p2", "f2p1", "f2p2")
CONTROLS = ("similarity", "double_fail", "regenerate", "outcome_only")
ARMS = ("interaction",) + CONTROLS
BRANCHES = PAIRS + ("double_fail", "regenerate")
SIGN = np.array([0.5, -0.5, -0.5, 0.5])
REPEATS = 2
TOKEN_CAP = 3072  # prompt prefill, plus up to 640 generated tokens
OUTPUT_CAP = 640


def similarity(left, right):
    return SequenceMatcher(None, left.split(), right.split(), autojunk=False).ratio()


def code_message(task):
    return ("Solve this ordinary algorithm problem. Return only a complete Python 3 program that reads stdin and writes "
            "stdout. Do not include prose.\n\n" + task["description"])


def prompt(tokenizer, task, initial=None, examples=()):
    message = code_message(task)
    if initial is not None:
        message += "\n\nRevise this previous program.\n```python\n" + initial + "\n```\n\nExecution feedback:\n"
        for status, given, expected, actual in examples:
            message += f"\n{status.upper()}\nInput:\n{given}\nExpected:\n{expected}\nActual:\n{actual}\n"
    return tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=True,
                                         add_generation_prompt=True)


def panel(task, initial, outcomes, tokenizer):
    observations = [(state, given, expected, actual) for ((given, expected), (state, actual)) in
                    zip(task["feedback"], outcomes, strict=True) if state in {"pass", "wrong"} and len(actual) <= 192]
    fails = [item for item in observations if item[0] == "wrong"]
    passes = [item for item in observations if item[0] == "pass"]
    if min(len(fails), len(passes)) < 2:
        return None
    # Include the best similarity pair in the matrix; choose the other witnesses
    # deterministically, independently of any repaired program or scoring labels.
    f1, p1 = max(((f, p) for f in fails for p in passes), key=lambda pair: similarity(pair[0][1], pair[1][1]))
    f2 = next(f for f in fails if f != f1)
    p2 = next(p for p in passes if p != p1)
    combinations = dict(zip(PAIRS, ((f1, p1), (f1, p2), (f2, p1), (f2, p2)), strict=True))
    combinations["double_fail"] = (f1, f2)
    lengths = {arm: len(prompt(tokenizer, task, initial, cases)) for arm, cases in combinations.items()}
    if max(lengths.values()) > TOKEN_CAP - OUTPUT_CAP:
        return None
    if max(lengths.values()) - min(lengths.values()) > 96:
        return None
    return {"examples": combinations,
            "features": [features(task, initial, *combinations[arm], lengths[arm]) for arm in PAIRS],
            "closest": "f1p1"}


def features(task, code, fail, passed, prompt_len):
    _, fi, fe, fa = fail
    _, pi, pe, _ = passed
    sim = similarity(fi, pi)
    output_similarity = similarity(fe, pe)
    same_shape = [len(line.split()) for line in fi.splitlines()] == [len(line.split()) for line in pi.splitlines()]
    return [len(task["description"]) / 4200, len(code) / 4000, code.count("\n") / 40, prompt_len / TOKEN_CAP,
            len(fi) / 192, len(pi) / 192, len(fe) / 192, len(pe) / 192, len(fa) / 192,
            similarity(fe, fa), sim, output_similarity, abs(len(fi) - len(pi)) / 192, float(same_shape),
            sim * (1 - output_similarity), sim * same_shape, abs(len(fe) - len(pe)) / 192]


def fit(rows, weight):
    x = np.array([row["features"] for row in rows])
    y = np.array([[average(row["branches"][arm], "success") for arm in PAIRS] for row in rows])
    center = x.mean(axis=(0, 1))
    scale = x.std(axis=(0, 1)).clip(0.01)
    design = design_matrix(x, center, scale)
    d_design = np.einsum("ijk,j->ik", design, SIGN)
    d_target = y @ SIGN
    design = design.reshape(-1, design.shape[-1]) / np.sqrt(y.size)
    target = y.ravel() / np.sqrt(y.size)
    size = design.shape[-1]
    penalty = np.eye(size) * 0.1
    penalty[0, 0] = 0
    normal = design.T @ design + penalty + weight * (d_design.T @ d_design) / len(x)
    rhs = design.T @ target + weight * (d_design.T @ d_target) / len(x)
    return {"center": center.tolist(), "scale": scale.tolist(), "coef": np.linalg.solve(normal, rhs).tolist()}


def design_matrix(x, center, scale):
    scaled = np.clip((np.asarray(x) - center) / scale, -5.0, 5.0)
    return np.concatenate((np.ones((*scaled.shape[:-1], 1)), scaled), axis=-1)


def choose(model, x):
    values = design_matrix(x, model["center"], model["scale"]) @ model["coef"]
    return PAIRS[int(values.argmax())]


def average(traces, key):
    values = [trace[key] for trace in traces if trace[key] is not None]
    return fmean(values) if values else None


def paired_intervals(rows, comparator_count):
    groups = defaultdict(list)
    for row in rows:
        groups[row["contest"]].append([row["interaction"] - row[arm] for arm in CONTROLS])
    totals = np.array([np.sum(values, axis=0) for values in groups.values()])
    sizes = np.array([len(values) for values in groups.values()])
    rng = np.random.default_rng(20261124)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)), size=4000)
    draws = (weights @ totals) / (weights @ sizes)[:, None]
    return {arm: np.quantile(draws[:, j], [0.025 / comparator_count, 1 - 0.025 / comparator_count]).tolist()
            for j, arm in enumerate(CONTROLS)}


def collect_scored(train, sealed):
    eligible_train = [row for row in train if row["eligible"]]
    fit_started = monotonic()
    models = {"outcome_only": fit(eligible_train, 0), "interaction": fit(eligible_train, 2)}
    fit_seconds = monotonic() - fit_started
    rows = []
    for row in sealed:
        started = monotonic()
        selected = {arm: choose(model, row["features"]) for arm, model in models.items()} if row["eligible"] else {}
        select_seconds = monotonic() - started
        selection = {**selected, "similarity": row.get("closest"), "double_fail": "double_fail",
                     "regenerate": "regenerate"}
        branches = {arm: row["branches"][selection[arm] if row["eligible"] else "regenerate"] for arm in ARMS}
        rows.append({"id": row["id"], "contest": row["contest"], "eligible": row["eligible"],
                     "selected": selection, "selection_seconds": select_seconds,
                     **{arm: average(traces, "success") for arm, traces in branches.items()},
                     "cost": {arm: {key: average(traces, key) for key in
                                    ("tokens", "prompt_tokens", "output_tokens", "executions", "judge_seconds",
                                     "generation_seconds", "retained", "repair_fraction")}
                              for arm, traces in branches.items()}})
    cohort = [row for row in rows if row["eligible"]]
    metrics = {}
    for name, cases in (("eligible", cohort), ("all_initial_failures", rows)):
        metrics[name] = {arm: {"pass_rate": fmean(row[arm] for row in cases),
                               **{key: average([row["cost"][arm] for row in cases], key)
                                  for key in cases[0]["cost"][arm]}} for arm in ARMS}
    rates, overall = metrics["eligible"], metrics["all_initial_failures"]
    best = max(CONTROLS, key=lambda arm: rates[arm]["pass_rate"])
    gain = rates["interaction"]["pass_rate"] - rates[best]["pass_rate"]
    intervals = paired_intervals(cohort, len(CONTROLS))
    all_gain = overall["interaction"]["pass_rate"] - overall[best]["pass_rate"]
    budget_ok = all(rates["interaction"]["tokens"] <= 1.03 * rates[arm]["tokens"]
                    for arm in (best, "outcome_only"))
    quality = {"five_points": gain >= 0.05, "positive_simultaneous_lower": min(ci[0] for ci in intervals.values()) > 0,
               "positive_all_failures": all_gain > 0}
    gates = {**quality, "matched_generation_tokens": budget_ok}
    verdict = "NO-GO" if not all(quality.values()) else "GO" if budget_ok else "INCOMPARABLE"
    return {"verdict": verdict,
            "strongest_baseline": best, "gain": gain, "all_failures_gain": all_gain, "intervals": intervals,
            "gates": gates, "metrics": metrics, "fit_seconds": fit_seconds,
            "select_seconds": sum(row["selection_seconds"] for row in rows),
            "selected_counts": {arm: dict(Counter(row["selected"][arm] for row in cohort)) for arm in models}}, models, rows


def order_key(task_id):
    return hashlib.sha256(("20261124:" + task_id).encode("utf-8")).hexdigest()


def job_seed(phase, rank, batch):
    return random.Random(f"{phase}:20261124:{rank}:{batch}").randrange(2**32)
