"""Answer canonicalization, voting and frozen P0a gates for the amplification-gap probe.

Pure standard library so the analysis can be re-run anywhere from the rank JSONL files.
"""

import math
import re
from collections import Counter
from fractions import Fraction

PROTOCOL = "amp_gap_p0a_v1"
LOW_ARM = "short"
HIGH_ARMS = ("force", "verify", "code", "think")
REPORT_TAUS = (0.5, 0.625, 0.75, 0.875, 1.0)
GATE_TAUS = (0.75, 0.875)
GAP_LOW_MAX = 0.5
RZERO_BAND = (0.25, 0.75)
G1_MIN_N, G1_PRECISION = 100, 0.90
G2_MIN_N, G2_PRECISION, G2_MARGIN = 30, 0.85, 0.10
G3_SHARE = 0.10
INFRA_MAX = 0.01

BOX = re.compile(r"\\(?:boxed|fbox)\s*")
NUMBER = re.compile(r"[+-]?\d+")
DECIMAL = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+)")
THOUSANDS = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
TEX_FRAC = re.compile(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}")
SLASH = re.compile(r"(-?\d+)/(-?\d+)")
TEXT_WRAP = re.compile(r"\\(?:text|textbf|mathrm|mbox|mathbf)\{([^{}]*)\}")
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces) or \\boxed x, else None."""
    if not text:
        return None
    matches = list(BOX.finditer(text))
    for match in reversed(matches):
        start = match.end()
        if start < len(text) and text[start] == "{":
            depth = 0
            for index in range(start, len(text)):
                char = text[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start + 1:index].strip()
            continue  # unbalanced (truncated) box: try an earlier one
        token = re.match(r"([^\s$]+)", text[start:])
        if token:
            return token.group(1).strip()
    return None


def _fix_fracs(string):
    pieces = string.split("\\frac")
    result = pieces[0]
    for piece in pieces[1:]:
        result += "\\frac"
        if piece.startswith("{"):
            result += piece
        elif len(piece) >= 2:
            a, b, rest = piece[0], piece[1], piece[2:]
            result += "{" + a + "}" + ("{" + b + "}" + rest if b != "{" else b + rest)
        else:
            return string
    return result


def _fix_sqrt(string):
    pieces = string.split("\\sqrt")
    result = pieces[0]
    for piece in pieces[1:]:
        if piece and piece[0] not in "{[":
            result += "\\sqrt{" + piece[0] + "}" + piece[1:]
        else:
            result += "\\sqrt" + piece
    return result


def strip_tex(string):
    """Hendrycks-style normalization of a final answer string."""
    string = string.strip().replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    for old, new in (("tfrac", "frac"), ("dfrac", "frac"), ("\\left", ""), ("\\right", ""),
                     ("\\displaystyle", ""), ("^{\\circ}", ""), ("^\\circ", ""), ("\\$", ""), ("$", ""),
                     ("\\%", ""), ("%", ""), ("\\,", ""), ("\\;", ""), ("\\:", ""), ("\\ ", ""), ("~", "")):
        string = string.replace(old, new)
    previous = None
    while previous != string:
        previous = string
        string = TEXT_WRAP.sub(lambda match: match.group(1), string)
    string = re.sub(r"\s*(degrees?|units?|cm|meters?|inches|feet|dollars?|cents?)\s*$", "", string)
    string = string.rstrip(".").strip()
    if string.startswith("."):
        string = "0" + string
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string.count("=") == 1 and len(string.split("=")[0].strip()) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    return string


def _python_to_tex(string):
    string = re.sub(r"sqrt\(([^()]*)\)", r"\\sqrt{\1}", string)
    string = re.sub(r"(?<![\\a-zA-Z])pi(?![a-zA-Z])", r"\\pi", string)
    return string.replace("**", "^").replace("*", "")


def _rational(string):
    if THOUSANDS.fullmatch(string):
        string = string.replace(",", "")
    if NUMBER.fullmatch(string):
        return Fraction(int(string))
    if DECIMAL.fullmatch(string):
        value = Fraction(string)
        decimals = len(string.split(".")[1])
        if decimals >= 6:
            approx = value.limit_denominator(1000)
            if abs(approx - value) < Fraction(1, 10**9):
                return approx
        return value
    match = TEX_FRAC.fullmatch(string)
    if match:
        sign, a, b = match.groups()
        if int(b) != 0:
            return Fraction(int(a), int(b)) * (-1 if sign else 1)
    match = SLASH.fullmatch(string)
    if match and int(match.group(2)) != 0:
        return Fraction(int(match.group(1)), int(match.group(2)))
    return None


def canonical(answer):
    """Voting key: `num:<reduced fraction>` for rationals, else `tex:<normalized string>`; None if empty."""
    if answer is None:
        return None
    string = strip_tex(str(answer))
    if not string:
        return None
    value = _rational(string)
    if value is None:
        value = _rational(strip_tex(_python_to_tex(string)))
    if value is not None:
        return f"num:{value}"
    return "tex:" + string


def code_answer(stdout):
    """Canonical answer from the last non-empty stdout line of an executed program."""
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[-1]
    boxed = last_boxed(line)
    return canonical(_python_to_tex(boxed if boxed is not None else line))


def extract_code(text):
    blocks = CODE_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks and blocks[-1].strip() else None


def gold_answer(row):
    """Canonical gold answer for MATH-style (solution) or GSM8K-style (#### answer) rows."""
    if row.get("answer") not in (None, ""):
        raw = str(row["answer"])
        if "####" in raw:
            raw = raw.split("####")[-1]
        return canonical(raw.strip())
    if row.get("solution"):
        return canonical(last_boxed(row["solution"]))
    return None


def vote(keys):
    """Mode over non-empty answers (ties: earliest first occurrence) and its share of all samples."""
    counts = Counter(key for key in keys if key is not None)
    if not counts:
        return None, 0.0
    first = {}
    for index, key in enumerate(keys):
        if key is not None and key not in first:
            first[key] = index
    mode = min(counts, key=lambda key: (-counts[key], first[key]))
    return mode, counts[mode] / len(keys)


def wilson_low(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre - margin) / (1 + z * z / n), 4)


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def problem_view(row):
    """Per-arm mode, agreement and correctness for one problem record."""
    view = {}
    for arm, data in row["arms"].items():
        answers = data["answers"]
        mode, agreement = vote(answers)
        view[arm] = {"mode": mode, "agreement": agreement, "mode_correct": mode is not None and mode == row["gold"],
                     "sample_correct": [answer is not None and answer == row["gold"] for answer in answers]}
    return view


def arm_summary(rows, views, arm):
    data = [row["arms"][arm] for row in rows]
    samples = sum(len(item["answers"]) for item in data)
    finishes = Counter(finish for item in data for finish in item["finish"])
    correct = sum(sum(view[arm]["sample_correct"]) for view in views)
    summary = {"problems": len(rows), "samples": samples, "pass_at_1": ratio(correct, samples),
               "majority_accuracy": ratio(sum(view[arm]["mode_correct"] for view in views), len(views)),
               "mean_agreement": ratio(sum(view[arm]["agreement"] for view in views), len(views)),
               "no_answer_rate": ratio(sum(answer is None for item in data for answer in item["answers"]), samples),
               "finish": dict(sorted(finishes.items())),
               "infra_rate": ratio(sum(v for f, v in finishes.items() if f.startswith("infra")), samples),
               "mean_output_tokens": ratio(sum(sum(item["tokens"]) for item in data), samples), "calibration": {}}
    for tau in REPORT_TAUS:
        chosen = [view for view in views if view[arm]["agreement"] >= tau - 1e-9]
        hits = sum(view[arm]["mode_correct"] for view in chosen)
        summary["calibration"][f"{tau:g}"] = {"n": len(chosen), "coverage": ratio(len(chosen), len(views)),
                                              "precision": ratio(hits, len(chosen)), "wilson_low": wilson_low(hits, len(chosen))}
    return summary


def gap_summary(rows, views, high, arm_stats):
    n = len(views)
    gaps = [view[high]["agreement"] - view[LOW_ARM]["agreement"] for view in views]
    lo, hi = RZERO_BAND
    band = [view for view in views if lo - 1e-9 <= view[LOW_ARM]["agreement"] <= hi + 1e-9]
    band_hits = sum(view[LOW_ARM]["mode_correct"] for view in band)
    band_precision = ratio(band_hits, len(band))
    infra_ok = (arm_stats[high]["infra_rate"] or 0) <= INFRA_MAX
    result = {"mean_gap": ratio(sum(gaps), n), "mean_positive_gap": ratio(sum(max(0.0, gap) for gap in gaps), n),
              "rzero_band": {"n": len(band), "low_label_precision": band_precision,
                             "wilson_low": wilson_low(band_hits, len(band))},
              "infra_ok": infra_ok, "by_tau": {}}
    for tau in GATE_TAUS:
        confident = [view for view in views if view[high]["agreement"] >= tau - 1e-9]
        confident_hits = sum(view[high]["mode_correct"] for view in confident)
        gap_set = [index for index, view in enumerate(views)
                   if view[LOW_ARM]["agreement"] <= GAP_LOW_MAX + 1e-9 and view[high]["agreement"] >= tau - 1e-9]
        gap_hits = sum(views[index][high]["mode_correct"] for index in gap_set)
        changed = sum(views[index][high]["mode"] != views[index][LOW_ARM]["mode"] for index in gap_set)
        band_high = [view for view in band if view[high]["agreement"] >= tau - 1e-9]
        band_high_hits = sum(view[high]["mode_correct"] for view in band_high)
        levels = Counter(rows[index]["level"] for index in gap_set)
        level_hits = Counter(rows[index]["level"] for index in gap_set if views[index][high]["mode_correct"])
        g1 = len(confident) >= G1_MIN_N and (confident_hits / len(confident)) >= G1_PRECISION
        gap_precision = gap_hits / len(gap_set) if gap_set else 0.0
        g2 = (len(gap_set) >= G2_MIN_N and gap_precision >= G2_PRECISION
              and band_precision is not None and gap_precision >= band_precision + G2_MARGIN)
        g3 = n > 0 and len(gap_set) / n >= G3_SHARE
        result["by_tau"][f"{tau:g}"] = {
            "high_confident": {"n": len(confident), "precision": ratio(confident_hits, len(confident)),
                               "wilson_low": wilson_low(confident_hits, len(confident))},
            "gap_set": {"n": len(gap_set), "share": ratio(len(gap_set), n), "precision": ratio(gap_hits, len(gap_set)),
                        "wilson_low": wilson_low(gap_hits, len(gap_set)),
                        "label_changed_vs_low_mode": ratio(changed, len(gap_set)),
                        "by_level": {str(level): {"n": count, "precision": ratio(level_hits[level], count)}
                                     for level, count in sorted(levels.items(), key=lambda item: str(item[0]))}},
            "rzero_band_with_high_label": {"n": len(band_high), "precision": ratio(band_high_hits, len(band_high))},
            "gates": {"G1_calibration": g1, "G2_gap_label_quality": g2, "G3_gap_share": g3, "infra_ok": infra_ok},
            "pass": bool(g1 and g2 and g3 and infra_ok)}
    return result


def analyze(rows):
    """Arm summaries, amplification gaps and the frozen P0a verdict."""
    if not rows:
        raise ValueError("no problem records")
    arms = list(rows[0]["arms"])
    if LOW_ARM not in arms or any(list(row["arms"]) != arms for row in rows):
        raise ValueError("every record must carry the same arms, including the low-budget arm")
    views = [problem_view(row) for row in rows]
    stats = {arm: arm_summary(rows, views, arm) for arm in arms}
    gaps = {arm: gap_summary(rows, views, arm, stats) for arm in arms if arm in HIGH_ARMS}
    passing = [{"arm": arm, "tau": tau} for arm, gap in gaps.items() for tau, cell in gap["by_tau"].items() if cell["pass"]]
    levels = Counter(row["level"] for row in rows)
    return {"protocol": PROTOCOL, "problems": len(rows), "levels": {str(k): v for k, v in sorted(levels.items(), key=lambda item: str(item[0]))},
            "arms": stats, "gaps": gaps, "passing_cells": passing, "verdict": "GO" if passing else "NO-GO",
            "frozen_gates": {"low_arm": LOW_ARM, "gate_taus": GATE_TAUS, "gap_low_max": GAP_LOW_MAX,
                             "rzero_band": RZERO_BAND, "G1": {"min_n": G1_MIN_N, "precision": G1_PRECISION},
                             "G2": {"min_n": G2_MIN_N, "precision": G2_PRECISION, "margin_over_band": G2_MARGIN},
                             "G3": {"min_share": G3_SHARE}, "infra_max": INFRA_MAX}}
