"""Knownness strata, the three kill-test readouts, bootstrap intervals and the frozen decision.

Frozen protocol: research/protocols/answer_prior_p0.md. Standard library only.
"""

import random
from collections import Counter, defaultdict

from work3.answer_prior_text import abstains, clean_answer, correct, norm_answer

STRATA = ("known", "wrong", "unknown", "partial")
CONFIDENT, SCATTERED = 6, 3          # of 8 samples
MIN_N = 300
T1_GAP = 0.20                        # dropout(treat) - dropout(control)
T2_GAP, T2_K = 0.08, 10              # matched recall@10, unknown - wrong
T3_KNOWN, T3_GAP = 0.65, 0.15        # preference for the original passage
RECALL_FLOOR, RECALL_K = 0.30, 20    # encoder sanity, answer recall@20 over all questions
ACCURACY_RANGE = (0.05, 0.70)        # closed-book greedy accuracy sanity
TOP = 5
BOOTSTRAP, BOOT_SEED = 1000, 20261127


def stratum(greedy, samples, alias_norms):
    """known / wrong (confidently) / unknown / partial from one greedy answer and eight samples."""
    answer = clean_answer(greedy)
    sampled = [clean_answer(text) for text in samples]
    greedy_correct = correct(answer, alias_norms)
    samples_correct = sum(correct(text, alias_norms) for text in sampled)
    greedy_abstains = abstains(answer)
    clusters = Counter(norm_answer(text) for text in sampled if not abstains(text))
    mode, size = clusters.most_common(1)[0] if clusters else ("", 0)
    if greedy_correct and samples_correct >= CONFIDENT:
        label = "known"
    elif (not greedy_correct and not greedy_abstains and samples_correct == 0 and size >= CONFIDENT
          and mode == norm_answer(answer)):
        label = "wrong"
    elif not greedy_correct and samples_correct <= 1 and (size <= SCATTERED or greedy_abstains):
        label = "unknown"
    else:
        label = "partial"
    return {"stratum": label, "answer": answer, "greedy_correct": greedy_correct,
            "samples_correct": samples_correct, "mode_size": size, "abstains": greedy_abstains}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def interval(values, low=0.025, high=0.975):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return [round(values[int(low * (len(values) - 1))], 4), round(values[int(high * (len(values) - 1))], 4)]


def bootstrap(units, statistic, seed_key):
    """Percentile interval of statistic over resampled units (questions or pairs)."""
    generator = random.Random(f"{BOOT_SEED}:{seed_key}")
    return interval(statistic([units[generator.randrange(len(units))] for _ in units]) for _ in range(BOOTSTRAP)) \
        if units else None


def rounded(value):
    return None if value is None else round(value, 4)


def pop_edges(values):
    values = sorted(value for value in values if value is not None)
    return [values[int(len(values) * q / 5)] for q in range(1, 5)] if values else []


def pop_bin(value, edges):
    if value is None:
        return -1
    return sum(value > edge for edge in edges)


# ----------------------------------------------------------------------------- T1 swap sensitivity

def t1(records):
    """records: {qid, stratum, rank_original, rank_treat, rank_control} with ranks re-scored among the top-20 pool."""
    eligible = [row for row in records if row["rank_original"] <= TOP]

    def gap(rows):
        if not rows:
            return None
        return mean(row["rank_treat"] > TOP for row in rows) - mean(row["rank_control"] > TOP for row in rows)

    by_stratum = {}
    for label in STRATA:
        rows = [row for row in eligible if row["stratum"] == label]
        by_stratum[label] = {"n": len(rows), "gap": rounded(gap(rows)),
                             "dropout_treat": rounded(mean(row["rank_treat"] > TOP for row in rows)),
                             "dropout_control": rounded(mean(row["rank_control"] > TOP for row in rows))}
    value = gap(eligible)
    passed = value is not None and len(eligible) >= MIN_N and value >= T1_GAP
    return {"n": len(eligible), "constructed": len(records), "lost_top5_on_rescore": len(records) - len(eligible),
            "dropout_treat": rounded(mean(row["rank_treat"] > TOP for row in eligible)),
            "dropout_control": rounded(mean(row["rank_control"] > TOP for row in eligible)),
            "gap": rounded(value), "ci95": bootstrap(eligible, gap, "t1"),
            "mean_rank_treat": rounded(mean(row["rank_treat"] for row in eligible)),
            "mean_rank_control": rounded(mean(row["rank_control"] for row in eligible)),
            "mean_score_drop_treat": rounded(mean(row["drop_treat"] for row in eligible)),
            "mean_score_drop_control": rounded(mean(row["drop_control"] for row in eligible)),
            "by_stratum": by_stratum, "threshold": T1_GAP, "min_n": MIN_N, "pass": passed}


# ----------------------------------------------------------------------------- T2 recall by knownness

def matched_gap(rows, k):
    """Recall(unknown) - recall(wrong), the unknown stratum reweighted to the wrong stratum's (relation, popularity) cells."""
    cells = defaultdict(lambda: {"unknown": [], "wrong": []})
    for row in rows:
        if row["stratum"] in ("unknown", "wrong"):
            cells[(row["prop"], row["pop_bin"])][row["stratum"]].append(row["hits"][k])
    total = weighted = 0.0
    for cell in cells.values():
        if cell["unknown"] and cell["wrong"]:
            weight = len(cell["wrong"])
            weighted += weight * (mean(cell["unknown"]) - mean(cell["wrong"]))
            total += weight
    return weighted / total if total else None


def t2(rows, gated):
    """rows: {qid, stratum, prop, pop_bin, hits: {k: bool}} for one retrieval system."""
    counts = Counter(row["stratum"] for row in rows)
    cells_wrong = Counter((row["prop"], row["pop_bin"]) for row in rows if row["stratum"] == "wrong")
    cells_unknown = {(row["prop"], row["pop_bin"]) for row in rows if row["stratum"] == "unknown"}
    matched_wrong = sum(count for cell, count in cells_wrong.items() if cell in cells_unknown)
    result = {"n": dict(counts), "matched_wrong": matched_wrong,
              "recall": {str(k): {label: rounded(mean(row["hits"][k] for row in rows if row["stratum"] == label))
                                  for label in STRATA} | {"all": rounded(mean(row["hits"][k] for row in rows))}
                         for k in (5, 10, 20)},
              "matched_gap": {str(k): rounded(matched_gap(rows, k)) for k in (5, 10, 20)}}
    if gated:
        value = matched_gap(rows, T2_K)
        units = [row for row in rows if row["stratum"] in ("unknown", "wrong")]
        result.update(ci95=bootstrap(units, lambda sample: matched_gap(sample, T2_K), "t2"), threshold=T2_GAP,
                      min_n=MIN_N)
        result["pass"] = (value is not None and counts["wrong"] >= MIN_N and counts["unknown"] >= MIN_N
                          and matched_wrong >= MIN_N and value >= T2_GAP)
    return result


def intrusion(rows):
    """Wrong stratum: top-10 contains the model's own wrong answer vs a placebo wrong answer of the same relation."""
    rows = [row for row in rows if row.get("intrusion_own") is not None and row.get("intrusion_placebo") is not None]
    excess = lambda sample: mean(row["intrusion_own"] - row["intrusion_placebo"] for row in sample) if sample else None
    return {"n": len(rows), "own": rounded(mean(row["intrusion_own"] for row in rows)),
            "placebo": rounded(mean(row["intrusion_placebo"] for row in rows)),
            "excess": rounded(excess(rows)), "ci95": bootstrap(rows, excess, "intrusion")}


# ----------------------------------------------------------------------------- T3 changed facts

def preference(original, edited):
    return 1.0 if original > edited else 0.0 if original < edited else 0.5


def t3(directions):
    """directions: {pair, stratum, pref}; both directions of each within-stratum cross-swapped pair."""
    by = {label: [row for row in directions if row["stratum"] == label] for label in ("known", "unknown", "wrong")}
    raw = {label: mean(row["pref"] for row in rows) for label, rows in by.items()}

    def clustered(label):
        pairs = defaultdict(list)
        for row in by[label]:
            pairs[row["pair"]].append(row["pref"])
        return list(pairs.values())

    known_pairs, unknown_pairs = clustered("known"), clustered("unknown")
    flat = lambda groups: mean(value for group in groups for value in group)
    generator = random.Random(f"{BOOT_SEED}:t3")
    gaps = []
    for _ in range(BOOTSTRAP if known_pairs and unknown_pairs else 0):
        known = [known_pairs[generator.randrange(len(known_pairs))] for _ in known_pairs]
        unknown = [unknown_pairs[generator.randrange(len(unknown_pairs))] for _ in unknown_pairs]
        gaps.append(flat(known) - flat(unknown))
    gap = None if raw["known"] is None or raw["unknown"] is None else raw["known"] - raw["unknown"]
    passed = (len(by["known"]) >= MIN_N and len(by["unknown"]) >= MIN_N and gap is not None
              and raw["known"] >= T3_KNOWN and gap >= T3_GAP)
    return {"n": {label: len(rows) for label, rows in by.items()},
            "pref_original": {label: rounded(value) for label, value in raw.items()}, "gap_known_unknown":
            rounded(gap), "ci95_gap": interval(gaps), "ci95_known": bootstrap(known_pairs, flat, "t3k"),
            "thresholds": {"known": T3_KNOWN, "gap": T3_GAP}, "min_n": MIN_N, "pass": passed}


def attraction(rows):
    """Wrong stratum: preference for the true passage against the model's own wrong answer (own) vs the same wrong
    answer placed in a partner's passage (placebo). Positive excess = the retriever leans to the model's belief."""
    excess = lambda sample: mean(row["placebo"] - row["own"] for row in sample) if sample else None
    return {"n": len(rows), "pref_vs_own_wrong": rounded(mean(row["own"] for row in rows)),
            "pref_vs_placebo_wrong": rounded(mean(row["placebo"] for row in rows)),
            "excess": rounded(excess(rows)), "ci95": bootstrap(rows, excess, "attraction")}


# ----------------------------------------------------------------------------- decision

def decide(retrievers, generator):
    """GO when some valid dense retriever passes at least two of T1-T3; INCOMPLETE when no retriever or the generator
    fails its sanity check."""
    low, high = ACCURACY_RANGE
    generator_ok = generator["accuracy"] is not None and low <= generator["accuracy"] <= high
    valid = {name: result for name, result in retrievers.items() if result["valid"]}
    passing = {name: [test for test in ("T1", "T2", "T3") if result[test]["pass"]] for name, result in valid.items()}
    if not generator_ok or not valid:
        verdict = "INCOMPLETE"
    elif any(len(tests) >= 2 for tests in passing.values()):
        verdict = "GO"
    else:
        verdict = "NO-GO"
    return {"verdict": verdict, "generator_ok": generator_ok, "valid_retrievers": sorted(valid),
            "passing_tests": passing}
