"""Frozen allocation protocol, paired decision rules and budget accounting."""

from math import comb, fsum
from random import Random

PROTOCOL = "wiki_rsi_feedback_allocation_v1"
DATASETS = ("hotpotqa", "2wikimultihopqa")
PHASES = ("lock", "prepare", "mechanism", "pilot", "confirm")
HOURS = {"lock": 0.75, "prepare": 2, "mechanism": 2, "pilot": 5, "confirm": 2}
SEEDS = (20261123, 20261124)
SPLITS = {"schedule": 96, "development": 250, "confirmation": 600}
MIN_CLUSTERS = {"schedule": 45, "development": 120, "confirmation": 300}
FAMILIES = ("person", "place", "work", "organization")
PROPOSALS = 4096
MIN_PAIRS = 640
MIN_FAMILY = 80
MAX_SHARE = 0.5
BATCH = 8
INPUT_LIMIT = 1840
ANSWER_TOKENS = 64
QUERY_TOKENS = 96
ACCUMULATION = 4
ROUNDS = 2
PROBE_STEPS = 16
CONTINUE_STEPS = 32
MECHANISM_STEPS = 32
TRAIN_STEPS = 2 * PROBE_STEPS + CONTINUE_STEPS
SIGNALS = ("search", "read")
FIXED = {"fixed0": 0, "fixed25": 1, "fixed50": 2, "fixed75": 3, "fixed100": 4}
ARMS = ("adaptive", *FIXED, "triage", "frozen_probe")
PROBE_MIN_GAIN = 0.02
PROBE_MAX_P = 0.10
MECHANISM_MIN_GAIN = 0.015
PILOT_MIN_GAIN = 0.03
CONFIRM_MIN_GAIN = 0.04
MAX_COST_RATIO = 1.05
MIN_FORMAT = 0.85


def paired_delta(first, second):
    if first.keys() != second.keys() or not first:
        raise ValueError("Paired arms must score the identical nonempty set of questions")
    differences = [first[key]["correct"] - second[key]["correct"] for key in sorted(first)]
    return differences, sum(differences) / len(differences)


def probe_choice(search, read):
    differences, gain = paired_delta(search, read)
    wins, losses = differences.count(1), differences.count(-1)
    discordant = wins + losses
    # One-sided tail for a direction chosen by the data: use the two-sided exact p.
    p = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2 ** discordant))
    choice = ("search" if gain > 0 else "read") if abs(gain) >= PROBE_MIN_GAIN and p <= PROBE_MAX_P else "mix"
    return {"choice": choice, "gain": gain, "wins": wins, "losses": losses, "p_exact": p}


def enough_pairs(counts):
    n = sum(counts.values())
    return (n >= MIN_PAIRS and sum(counts.get(f, 0) >= MIN_FAMILY for f in FAMILIES) >= 3
            and max(counts.values(), default=0) <= MAX_SHARE * n)


def mechanism_gate(reports):
    effects = {}
    for seed in SEEDS:
        effects[str(seed)] = {
            state: paired_delta(reports[seed, state, "search"]["scores"],
                                reports[seed, state, "read"]["scores"])[1]
            for state in ("early", "late")
        }
    values = list(effects.values())
    ready = all(abs(row["early"]) >= MECHANISM_MIN_GAIN and abs(row["late"]) >= MECHANISM_MIN_GAIN
                and row["early"] * row["late"] < 0 and row["early"] * values[0]["early"] > 0
                for row in values)
    return {"effects": effects, "ready": ready}


def end_to_end(reports, fixed=None, *, final=False):
    if final and fixed is None:
        raise ValueError("Confirmation requires the fixed baseline selected on development")
    if fixed is None:
        fixed = max(FIXED, key=lambda arm: (
            fsum(row["correct"] for seed in SEEDS for row in reports[seed, arm]["scores"].values()), arm))
    comparisons = {}
    for arm in (fixed, "triage", "frozen_probe"):
        effects, cost, by_dataset = {}, {}, {ds: [] for ds in DATASETS}
        by_question = {}
        for seed in SEEDS:
            a, b = reports[seed, "adaptive"], reports[seed, arm]
            scores_a, scores_b = a["scores"], b["scores"]
            _, effect = paired_delta(scores_a, scores_b)
            effects[str(seed)] = effect
            cost[str(seed)] = a["training_seconds"] / b["training_seconds"]
            for key in sorted(scores_a):
                qa, qb = scores_a[key], scores_b[key]
                if qa["dataset"] != qb["dataset"] or qa["cluster"] != qb["cluster"]:
                    raise ValueError("Evaluation provenance differs across arms")
                difference = qa["correct"] - qb["correct"]
                by_dataset[qa["dataset"]].append(difference)
                if key not in by_question:
                    by_question[key] = (qa["dataset"], qa["cluster"], [])
                dataset, cluster, values = by_question[key]
                if (dataset, cluster) != (qa["dataset"], qa["cluster"]):
                    raise ValueError("Evaluation provenance differs across seeds")
                values.append(difference)
        if any(len(values) != len(SEEDS) for _, _, values in by_question.values()):
            raise ValueError("Evaluation questions differ across seeds")
        average = fsum(effects.values()) / len(SEEDS)
        ci = clustered_interval(by_question) if final else None
        dataset_gain = {ds: fsum(ds_effects) / len(ds_effects) for ds, ds_effects in by_dataset.items()}
        ready = (min(effects.values()) > 0 and average >= (CONFIRM_MIN_GAIN if final else PILOT_MIN_GAIN)
                 and max(cost.values()) <= MAX_COST_RATIO
                 and (not final or (ci[0] > 0 and min(dataset_gain.values()) >= 0)))
        comparisons[arm] = {"mean_gain": average, "by_seed": effects, "by_dataset": dataset_gain,
                            "cost_ratio": cost, "ci95": ci, "ready": ready}
    switches = any(len(set(reports[seed, "adaptive"]["choices"]) & set(SIGNALS)) == 2 for seed in SEEDS)
    base = {str(seed): paired_delta(reports[seed, "adaptive"]["scores"], reports[SEEDS[0], "base"]["scores"])[1]
            for seed in SEEDS}
    valid_format = all(
        fsum(case["valid"] for case in row["scores"].values()) / len(row["scores"]) >= MIN_FORMAT
        for (seed, arm), row in reports.items() if arm in {"adaptive", fixed, "triage", "frozen_probe"}
    )
    return {"chosen_fixed": fixed, "comparisons": comparisons, "switch_observed": switches,
            "valid_format": valid_format, "vs_frozen_r3": base,
            "ready": switches and valid_format and min(base.values()) > 0
            and all(row["ready"] for row in comparisons.values())}


def clustered_interval(by_question):
    groups = {ds: {} for ds in DATASETS}
    for dataset, cluster, values in by_question.values():
        group = groups[dataset].setdefault(cluster, [0.0, 0])
        group[0] += fsum(values) / len(values)
        group[1] += 1
    groups = [list(members.values()) for members in groups.values()]
    if any(not group for group in groups):
        raise ValueError("Both frozen datasets must be represented")
    rng = Random(20261123)
    means = []
    for _ in range(4000):
        samples = [item for group in groups for item in rng.choices(group, k=len(group))]
        means.append(fsum(total for total, _ in samples) / sum(n for _, n in samples))
    means.sort()
    return [means[99], means[3899]]
