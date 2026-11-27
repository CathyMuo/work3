"""Statistics of the plan-B first experiment (retrieve-to-certify): FDR-certified thresholds, adaptive two-stage
retrieval policies, cost-matched retention, judge-label calibration and the frozen decision.

Frozen protocol: research/protocols/claim_cert_b0.md. Standard library only, so every rule is unit-tested on CPU.
"""

import hashlib
import math
import random
from collections import Counter

PROTOCOL = "claim_cert_b0_v1"
ALPHAS = (0.05, 0.1, 0.2)
DELTA = 0.1
# 50 fixed thresholds, evenly spaced in logit space from -4 to 10 (0.018 .. 0.99995); fixed before any data
GRID = tuple(1.0 / (1.0 + math.exp(-(-4.0 + 14.0 * index / 49))) for index in range(50))
CONFIGS = ("q1", "q3", "q5", "q10", "q20", "qa3", "qa10", "qa20")
COST = {"q1": 1, "q3": 3, "q5": 5, "q10": 10, "q20": 20, "qa3": 3, "qa10": 10, "qa20": 20}
STAGE1 = ("q1", "q3", "qa3")
STAGE2 = ("q10", "q20", "qa10", "qa20")
RULES = ("near", "below")
TARGETS = (5, 10)
SPLITS = {"run": 50, "smoke": 3}
FRACTIONS = {"tune": 0.25, "calib": 0.25}          # rest is test
G1_GAIN, G1_ALPHA = 0.10, 0.1
J_RATIO, J_FP_GAP = 1.5, 0.10
JUDGE_CUT = 0.5
BOOTSTRAP, BOOT_SEED = 1000, 20261128


# ----------------------------------------------------------------------------- certified thresholds

def logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    top = max(a, b)
    return top + math.log1p(math.exp(-abs(a - b)))


def binom_cdf(f, m, p):
    """P(Binomial(m, p) <= f), summed in log space (m up to tens of thousands)."""
    if f < 0:
        return 0.0
    if f >= m:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    term = m * log_q
    total = term
    for j in range(f):
        term += math.log((m - j) / (j + 1)) + log_p - log_q
        total = logaddexp(total, term)
    return min(1.0, math.exp(total))


def ltt(scores, truth, alpha, delta=DELTA, grid=GRID):
    """Lowest grid threshold whose false-discovery rate among retained claims is certified <= alpha.

    Each grid threshold is a hypothesis H0: P(false | retained) > alpha, tested by an exact binomial test conditional
    on the number retained; Bonferroni over the fixed grid gives P(the chosen threshold has FDR > alpha) <= delta.
    Returns None when no threshold is certified (retain nothing)."""
    pairs = sorted(zip(scores, truth), key=lambda item: -item[0])
    level = delta / len(grid)
    chosen, index, kept, false = None, 0, 0, 0
    for threshold in sorted(grid, reverse=True):
        while index < len(pairs) and pairs[index][0] >= threshold:
            kept += 1
            false += not pairs[index][1]
            index += 1
        # false > kept * alpha puts the observed count above the null mean: the p-value exceeds 0.5 > level
        if kept and false <= kept * alpha and binom_cdf(false, kept, alpha) <= level:
            chosen = threshold
    return chosen


def evaluate(scores, truth, threshold):
    """Retention of true claims, share of claims kept and realised FDR at a threshold (None keeps nothing)."""
    kept = [label for score, label in zip(scores, truth) if threshold is not None and score >= threshold]
    true_total = sum(truth)
    return {"tpr": sum(kept) / true_total if true_total else None, "kept": len(kept) / len(scores) if scores else 0.0,
            "fdr": 1.0 - sum(kept) / len(kept) if kept else 0.0}


# ----------------------------------------------------------------------------- adaptive two-stage policy

def logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def distance(score, centre, rule):
    value = logit(score)
    if rule == "near":
        return abs(value - centre)
    return centre - value if value < centre else math.inf


def fit_rule(stage1, truth, alpha, fraction, rule):
    """Escalation rule fixed on the tuning split: send the `fraction` of claims whose stage-1 score is closest to the
    stage-1 certified threshold ("near"), or closest from below ("below"), to the second stage."""
    threshold = ltt(stage1, truth, alpha)
    centre = logit(threshold if threshold is not None else max(GRID))
    gaps = sorted(distance(score, centre, rule) for score in stage1)
    finite = [gap for gap in gaps if gap != math.inf]
    if not finite:
        return {"centre": centre, "width": -1.0, "rule": rule}  # nothing can be escalated under this rule
    count = min(len(gaps), max(1, math.ceil(fraction * len(gaps))))
    # "below" never escalates claims above the stage-1 threshold, even when that leaves budget unused
    return {"centre": centre, "width": min(gaps[count - 1], finite[-1]), "rule": rule}


def escalated(score, fitted):
    return distance(score, fitted["centre"], fitted["rule"]) <= fitted["width"]


def final_scores(stage1, stage2, fitted):
    flags = [escalated(score, fitted) for score in stage1]
    return [second if flag else first for first, second, flag in zip(stage1, stage2, flags)], flags


def policy_cost(first, second, flags):
    return COST[first] + (sum(flags) / len(flags) if flags else 0.0) * COST[second]


# ----------------------------------------------------------------------------- one split

def split_of(qid, repeat):
    value = int(hashlib.sha256(f"{PROTOCOL}:split:{repeat}:{qid}".encode()).hexdigest()[:8], 16) / 2 ** 32
    if value < FRACTIONS["tune"]:
        return "tune"
    return "calib" if value < FRACTIONS["tune"] + FRACTIONS["calib"] else "test"


def column(claims, key):
    return [claim["scores"][key] for claim in claims]


def run_split(claims, repeat, label_key="truth"):
    """Uniform and adaptive policies on one tuning / calibration / test split.

    `label_key` names the labels used for tuning and calibration ("truth" or a judge label); the test split is
    always scored against the truth."""
    parts = {"tune": [], "calib": [], "test": []}
    for claim in claims:
        parts[split_of(claim["qid"], repeat)].append(claim)
    tune, calib, test = parts["tune"], parts["calib"], parts["test"]
    labels = lambda rows: [claim[label_key] for claim in rows]
    truth_test = [claim["truth"] for claim in test]
    result = {"repeat": repeat, "sizes": {name: len(rows) for name, rows in parts.items()}, "uniform": {},
              "adaptive": {}}
    pooled = tune + calib  # the uniform baseline gets the tuning claims too (conservative for the method)
    for alpha in ALPHAS:
        uniform = {}
        for key in CONFIGS:
            threshold = ltt(column(pooled, key), labels(pooled), alpha)
            uniform[key] = {**evaluate(column(test, key), truth_test, threshold), "cost": COST[key],
                            "threshold": threshold}
        result["uniform"][str(alpha)] = uniform
        adaptive = {}
        for target in TARGETS:
            if not tune or not calib or not test:
                adaptive[str(target)] = {"tpr": None, "fdr": 0.0, "kept": 0.0, "cost": None, "first": None,
                                         "second": None, "rule": None, "threshold": None, "escalated": 0.0,
                                         "best_uniform": None, "gain": None}
                continue
            best = None
            for first in STAGE1:
                for second in STAGE2:
                    fraction = (target - COST[first]) / COST[second]
                    if fraction <= 0:
                        continue
                    fraction = min(1.0, fraction)
                    for rule in RULES:
                        fitted = fit_rule(column(tune, first), labels(tune), alpha, fraction, rule)
                        scores, flags = final_scores(column(tune, first), column(tune, second), fitted)
                        tuned = evaluate(scores, labels(tune), ltt(scores, labels(tune), alpha))
                        key = (tuned["tpr"] or 0.0, -policy_cost(first, second, flags), first, second, rule)
                        if best is None or key > best[0]:
                            best = (key, first, second, rule, fitted)
            _, first, second, rule, fitted = best
            calib_scores, _ = final_scores(column(calib, first), column(calib, second), fitted)
            threshold = ltt(calib_scores, labels(calib), alpha)
            test_scores, flags = final_scores(column(test, first), column(test, second), fitted)
            outcome = evaluate(test_scores, truth_test, threshold)
            cost = policy_cost(first, second, flags)
            affordable = [row["tpr"] or 0.0 for row in uniform.values() if row["cost"] <= cost + 1e-9]
            reference = max(affordable) if affordable else 0.0
            best_uniform = max((key for key, row in uniform.items() if row["cost"] <= cost + 1e-9),
                               key=lambda key: uniform[key]["tpr"] or 0.0, default=None)
            adaptive[str(target)] = {**outcome, "cost": cost, "first": first, "second": second, "rule": rule,
                                     "threshold": threshold, "escalated": sum(flags) / len(flags) if flags else 0.0,
                                     "best_uniform": best_uniform, "gain": (outcome["tpr"] or 0.0) - reference}
        result["adaptive"][str(alpha)] = adaptive
    return result


# ----------------------------------------------------------------------------- aggregation and readouts

def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))] if values else None


def rounded(value):
    return None if value is None else round(value, 4)


def aggregate(splits):
    rows = []
    for alpha in ALPHAS:
        key = str(alpha)
        uniform = {config: {"tpr": rounded(mean(split["uniform"][key][config]["tpr"] for split in splits)),
                            "fdr": rounded(mean(split["uniform"][key][config]["fdr"] for split in splits)),
                            "kept": rounded(mean(split["uniform"][key][config]["kept"] for split in splits))}
                   for config in CONFIGS}
        for target in TARGETS:
            cells = [split["adaptive"][key][str(target)] for split in splits]
            gains = [cell["gain"] for cell in cells]
            rows.append({"alpha": alpha, "target": target, "mean_gain": rounded(mean(gains)),
                         "gain_p10": rounded(percentile(gains, 0.1)), "gain_p90": rounded(percentile(gains, 0.9)),
                         "adaptive_tpr": rounded(mean(cell["tpr"] for cell in cells)),
                         "adaptive_fdr": rounded(mean(cell["fdr"] for cell in cells)),
                         "fdr_over_alpha_share": rounded(mean(float(cell["fdr"] > alpha) for cell in cells)),
                         "adaptive_cost": rounded(mean(cell["cost"] for cell in cells)),
                         "escalated": rounded(mean(cell["escalated"] for cell in cells)),
                         "policies": dict(Counter(f"{c['first']}->{c['second']}:{c['rule']}" for c in cells)),
                         "best_uniform": dict(Counter(str(cell["best_uniform"]) for cell in cells)),
                         "uniform": uniform})
    return rows


def auc(scores, truth):
    """Probability that a true claim outscores a false one (ties count half)."""
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks, index = [0.0] * len(scores), 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        for position in range(index, end + 1):
            ranks[order[position]] = (index + end) / 2 + 1
        index = end + 1
    positives = sum(truth)
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank for rank, label in zip(ranks, truth) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def judge_errors(claims, label_key):
    """False-positive rate of judge labels on false claims by knownness stratum, false-negative rate on true claims."""
    def fp(stratum):
        rows = [claim[label_key] for claim in claims if not claim["truth"] and claim["stratum"] == stratum]
        return (sum(rows) / len(rows) if rows else None), len(rows)

    wrong, n_wrong = fp("wrong")
    unknown, n_unknown = fp("unknown")
    partial, n_partial = fp("partial")
    trues = [claim[label_key] for claim in claims if claim["truth"]]
    falses = [claim for claim in claims if not claim["truth"] and claim["stratum"] in ("wrong", "unknown")]
    generator = random.Random(f"{BOOT_SEED}:{label_key}")
    wrong_rows = [claim[label_key] for claim in falses if claim["stratum"] == "wrong"]
    unknown_rows = [claim[label_key] for claim in falses if claim["stratum"] == "unknown"]
    gaps = []
    for _ in range(BOOTSTRAP if wrong_rows and unknown_rows else 0):
        a = [wrong_rows[generator.randrange(len(wrong_rows))] for _ in wrong_rows]
        b = [unknown_rows[generator.randrange(len(unknown_rows))] for _ in unknown_rows]
        gaps.append(sum(a) / len(a) - sum(b) / len(b))
    gap = None if wrong is None or unknown is None else wrong - unknown
    return {"fp_wrong": rounded(wrong), "fp_unknown": rounded(unknown), "fp_partial": rounded(partial),
            "n": {"wrong": n_wrong, "unknown": n_unknown, "partial": n_partial, "true": len(trues)},
            "fn_true": rounded(1 - sum(trues) / len(trues)) if trues else None,
            "fp_gap_wrong_unknown": rounded(gap),
            "ci95": [rounded(percentile(gaps, 0.025)), rounded(percentile(gaps, 0.975))] if gaps else None,
            "agreement": rounded(mean(float(claim[label_key] == claim["truth"]) for claim in claims))}


def judge_calibration(judged_splits):
    """Realised test FDR (against the truth) when tuning and calibration use judge labels instead of the truth."""
    rows = {}
    for alpha in ALPHAS:
        key = str(alpha)
        q3 = [split["uniform"][key]["q3"]["fdr"] for split in judged_splits]
        adaptive = {str(target): [split["adaptive"][key][str(target)]["fdr"] for split in judged_splits]
                    for target in TARGETS}
        rows[key] = {"uniform_q3_fdr": rounded(mean(q3)), "uniform_q3_ratio": rounded(mean(q3) / alpha),
                     **{f"adaptive_{target}_fdr": rounded(mean(values)) for target, values in adaptive.items()}}
    return rows


def decide(g1_rows, self_judge, validity):
    """GO: adaptive retrieval keeps >= 10 points more true claims at alpha 0.1 and cost 5 or 10, with mean test FDR
    within alpha. PIVOT: not GO, but self-judge calibration breaks the guarantee (realised FDR >= 1.5 alpha) and
    judge false positives concentrate on confident errors (>= 10 points over non-confident ones). Else NO-GO."""
    if not all(validity.values()):
        return {"verdict": "INCOMPLETE", "g1_pass": None, "judge_pass": None}
    g1 = [row for row in g1_rows if row["alpha"] == G1_ALPHA and row["target"] in TARGETS]
    g1_pass = any(row["mean_gain"] is not None and row["mean_gain"] >= G1_GAIN and row["adaptive_fdr"] is not None
                  and row["adaptive_fdr"] <= G1_ALPHA for row in g1)
    ratio = self_judge["calibration"][str(G1_ALPHA)]["uniform_q3_ratio"]
    gap = self_judge["errors"]["fp_gap_wrong_unknown"]
    judge_pass = ratio is not None and gap is not None and ratio >= J_RATIO and gap >= J_FP_GAP
    verdict = "GO" if g1_pass else "PIVOT" if judge_pass else "NO-GO"
    return {"verdict": verdict, "g1_pass": g1_pass, "judge_pass": judge_pass}
