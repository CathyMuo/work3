"""CPU-only paired analysis of retrieval in the saved Dr. Zero P0 evaluation dumps."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from work3.evidence_ref import DATASETS as SUPPORT_DATASETS
from work3.evidence_ref import answer_keys, contains, question_key, sha256, text_key, title_key
from work3.p0_diag import MULTI, question_of
from work3.p0_gate import FLASHRAG_COUNTS, SOURCE_PREFIX, resolve_dump

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "research/protocols/P0_retrieval_diagnosis.md"
BUDGETS = (1, 2, 3, 5)
TOOL = re.compile(r"<(tool_call|tool_response)>(.*?)</\1>", re.DOTALL)
TOOL_TAG = re.compile(r"</?tool_(?:call|response)>")
DOC = re.compile(r"(?m)^Doc (\d+) \(Title: (.+)\)[ \t]*$")
EMPTY = "No search results found."
ERROR_STARTS = ("Search error:", "Search request failed", "Error", "Unknown API state")


class TraceError(ValueError):
    """Logged tool interactions cannot be mapped to query ranks."""


@dataclass
class Trace:
    queries: list[list[tuple[str, str]]]
    first_call: int
    calls: int
    errors: int


def documents(block: str) -> list[tuple[str, str]]:
    if not block.strip():
        return []
    headers = list(DOC.finditer(block))
    if not headers or block[:headers[0].start()].strip():
        raise TraceError("document_header")
    passages = []
    for index, header in enumerate(headers, 1):
        if int(header[1]) != index:
            raise TraceError("document_numbering")
        end = headers[index].start() if index < len(headers) else len(block)
        passages.append((title_key(header[2]), text_key(block[header.end():end])))
    return passages


def trace_of(output: str) -> Trace:
    output = output.replace("\r\n", "\n")
    events = list(TOOL.finditer(output))
    if len(TOOL_TAG.findall(output)) != 2 * len(events):
        raise TraceError("unbalanced_tags")
    trace = Trace([], 0, 0, 0)
    pending = []
    for event in events:
        try:
            payload = json.loads(event[2])
        except ValueError as exc:
            raise TraceError("invalid_tool_json") from exc
        if not isinstance(payload, dict):
            raise TraceError("invalid_tool_payload")
        if event[1] == "tool_call":
            arguments = payload.get("arguments")
            queries = arguments.get("query_list") if isinstance(arguments, dict) else None
            if (payload.get("name") != "search" or not isinstance(queries, list) or not queries
                    or not all(isinstance(q, str) and q.strip() for q in queries)):
                raise TraceError("invalid_search_call")
            pending.append(len(queries))
            trace.calls += 1
            if trace.calls == 1:
                trace.first_call = len(queries)
            continue
        if not pending:
            raise TraceError("response_without_call")
        count = pending.pop(0)
        result = payload.get("result")
        if not isinstance(result, str):
            raise TraceError("missing_result_field")
        if result == EMPTY or result.startswith(ERROR_STARTS):
            trace.errors += int(result != EMPTY)
            trace.queries.extend([] for _ in range(count))
            continue
        blocks = re.split(r"\n[ \t]*---[ \t]*\n", result.replace("\r\n", "\n"))
        if len(blocks) != count:
            raise TraceError("query_block_count")
        trace.queries.extend(documents(block) for block in blocks)
    if pending:
        raise TraceError("call_without_response")
    return trace


def dataset_name(source: str) -> str:
    return source.removeprefix(SOURCE_PREFIX)


def test_questions(path: Path):
    from pyarrow.parquet import ParquetFile

    parquet = ParquetFile(path)
    names = parquet.schema_arrow.names
    columns = ["data_source", "reward_model"]
    columns += [name for name in ("question", "extra_info", "prompt") if name in names]
    rows = {ds: {} for ds in MULTI}
    counts = Counter()
    duplicate = {ds: set() for ds in MULTI}
    for batch in parquet.iter_batches(columns=columns):
        for row in batch.to_pylist():
            ds = dataset_name(row["data_source"])
            if ds not in rows:
                continue
            counts[ds] += 1
            question = row.get("question") or (row.get("extra_info") or {}).get("question")
            if not question:
                prompt = row.get("prompt") or []
                question = next((question_of(m["content"]) for m in reversed(prompt)
                                 if m.get("role") == "user"), "")
            if not question:
                raise ValueError(f"{ds}: no question in test parquet row {counts[ds]}")
            key = question_key(question)
            if key in rows[ds]:
                duplicate[ds].add(key)
            answer = (row.get("reward_model") or {}).get("ground_truth")
            if isinstance(answer, dict):
                answer = answer.get("target")
            answers = [answer] if isinstance(answer, str) else (answer or [])
            rows[ds][key] = [str(value) for value in answers if value is not None]
    for ds, keys in duplicate.items():
        for key in keys:
            del rows[ds][key]
    return rows, counts


@dataclass
class Observation:
    score: int
    calls: int
    queries: int
    first_call: int
    errors: int
    # Each rank is one-based. None denotes no hit; an absent key denotes no label.
    hit: dict[str, int | None]
    titles: dict[str, int]
    required_titles: int | None


def observe(row: dict, aliases: list[str], ref: dict | None) -> Observation:
    trace = trace_of(row["output"])
    score = float(row["score"])
    if score not in (0, 1):
        raise ValueError("P0 dump contains a non-binary exact-match score")
    phrases = {key for alias in aliases if (key := text_key(alias))}
    facts = ref["titles"] if ref else {}
    have_sentences = bool(facts) and all(s and "" not in s for s in facts.values())
    required_sentences = sum(map(len, facts.values()))
    hit = {"answer_string": None} if phrases else {}
    titles = {}
    sentences = set()
    if ref:
        hit.update(any_title=None, all_titles=None)
        if have_sentences:
            hit["all_sentences"] = None
        if ref["bridge_title"] is not None:
            hit.update(bridge_title=None, answer_title=None)
    for number, passages in enumerate(trace.queries, 1):
        for title, body in passages:
            if phrases and hit["answer_string"] is None:
                normalized_title = text_key(title)
                if any(contains(body, phrase) or contains(normalized_title, phrase) for phrase in phrases):
                    hit["answer_string"] = number
            if title in facts:
                titles.setdefault(title, number)
                if have_sentences:
                    sentences.update((title, sent) for sent in facts[title] if contains(body, sent))
        if ref:
            if titles and hit["any_title"] is None:
                hit["any_title"] = number
            if len(titles) == len(facts) and hit["all_titles"] is None:
                hit["all_titles"] = number
            if have_sentences and len(sentences) == required_sentences and hit["all_sentences"] is None:
                hit["all_sentences"] = number
    if "bridge_title" in hit:
        for role in ("bridge_title", "answer_title"):
            hit[role] = titles.get(ref[role])
    return Observation(int(score), trace.calls, len(trace.queries), trace.first_call, trace.errors,
                       hit, titles, len(facts) if ref else None)


def load_run(path: Path, tests: dict, references: dict):
    rows = {ds: {} for ds in MULTI}
    counts = {ds: Counter() for ds in MULTI}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            ds = dataset_name(item["data_source"])
            if ds not in rows:
                continue
            counts[ds]["dump_rows"] += 1
            key = question_key(question_of(item["input"]))
            if key not in tests[ds]:
                counts[ds]["unmatched_or_duplicate_test_question"] += 1
                continue
            if key in rows[ds]:
                counts[ds]["duplicate_dump_question"] += 1
                rows[ds][key] = None
                continue
            try:
                rows[ds][key] = observe(item, tests[ds][key], references.get(ds, {}).get(key))
            except TraceError as exc:
                counts[ds][f"trace_{exc}"] += 1
                rows[ds][key] = None
    return ({ds: {key: value for key, value in per_ds.items() if value is not None}
             for ds, per_ds in rows.items()}, counts)


def reached(item: Observation, metric: str, budget: int) -> bool:
    hit = item.hit[metric]
    return hit is not None and hit <= budget


def ratio(count: int, total: int) -> float | None:
    return count / total if total else None


METRICS = ("answer_string", "all_titles", "all_sentences", "bridge_title", "answer_title")


def summarize(rows: list[Observation]) -> dict:
    total = len(rows)
    annotated = [item for item in rows if item.required_titles is not None]
    result = {"n": total, "em": ratio(sum(x.score for x in rows), total),
              "calls_per_question": ratio(sum(x.calls for x in rows), total),
              "queries_per_question": ratio(sum(x.queries for x in rows), total),
              "search_error_questions": sum(x.errors > 0 for x in rows),
              "reached_query": {str(k): ratio(sum(x.queries >= k for x in rows), total) for k in BUDGETS},
              "support_titles": {
                  "n": len(annotated),
                  "recall_first_call": ratio(sum(sum(r <= x.first_call for r in x.titles.values())
                                                / x.required_titles for x in annotated), len(annotated)),
                  "recall_final": ratio(sum(len(x.titles) / x.required_titles for x in annotated), len(annotated)),
                  "recall_by_queries": {str(k): ratio(sum(sum(r <= k for r in x.titles.values())
                                                             / x.required_titles for x in annotated), len(annotated))
                                        for k in BUDGETS},
                  "mean_new_after_first_call": ratio(sum(sum(r > x.first_call for r in x.titles.values())
                                                         for x in annotated), len(annotated)),
              }}
    for metric in METRICS:
        group = [x for x in rows if metric in x.hit]
        n = len(group)
        first_call = sum(reached(x, metric, x.first_call) for x in group)
        final = sum(reached(x, metric, x.queries) for x in group)
        result[metric] = {
            "n": n, "first_call": ratio(first_call, n),
            "new_after_first_call": ratio(final - first_call, n),
            "final": ratio(final, n),
            "by_queries": {str(k): ratio(sum(reached(x, metric, k) for x in group), n) for k in BUDGETS},
        }
    return result


def counts_of(pairs, metric: str, budget: int) -> tuple[int, int, int, int]:
    present = [(reached(a, metric, budget), reached(b, metric, budget))
               for a, b in pairs if metric in a.hit and metric in b.hit]
    early = sum(a for a, _ in present)
    late = sum(b for _, b in present)
    losses = sum(a and not b for a, b in present)
    return (len(present), early, late, losses)


def paired_stats(counts) -> dict:
    n, early, late, lost = counts
    return {"n": n, "early": ratio(early, n), "late": ratio(late, n), "delta": ratio(late - early, n),
            "lost": lost, "gained": lost + late - early}


def macro_stats(groups: dict) -> dict | None:
    if any(not groups.get(ds, {}).get("n") for ds in SUPPORT_DATASETS):
        return None
    rng = np.random.default_rng(20261117)
    bootstrap = []
    for ds in SUPPORT_DATASETS:
        stats = groups[ds]
        n = stats["n"]
        lost, gained = stats["lost"], stats["gained"]
        draw = rng.multinomial(n, np.array([lost, gained, n - lost - gained]) / n, size=5000)
        bootstrap.append((draw[:, 1] - draw[:, 0]) / n)
    ci = np.percentile(np.mean(bootstrap, axis=0), [2.5, 97.5])
    return {"delta": mean(groups[ds]["delta"] for ds in SUPPORT_DATASETS),
            "ci95": ci.tolist()}


def compare_runs(early: dict, late: dict) -> dict:
    result = {}
    for ds in MULTI:
        pair = [(early[ds][key], late[ds][key]) for key in sorted(early[ds].keys() & late[ds].keys())]
        reached_two = [(a, b) for a, b in pair if min(a.queries, b.queries) >= 2]
        bridge_miss = [(a, b) for a, b in pair
                       if min(a.queries, b.queries) >= 1 and "bridge_title" in a.hit and "bridge_title" in b.hit
                       and not reached(a, "bridge_title", 1) and not reached(b, "bridge_title", 1)]
        result[ds] = {
            "paired": len(pair),
            "all_titles_q1": paired_stats(counts_of(pair, "all_titles", 1)),
            "all_titles_q2": paired_stats(counts_of(pair, "all_titles", 2)),
            "all_titles_q2_both_reached": paired_stats(counts_of(reached_two, "all_titles", 2)),
            "all_sentences_q2": paired_stats(counts_of(pair, "all_sentences", 2)),
            "bridge_joint_q1_miss_q3": paired_stats(counts_of(bridge_miss, "bridge_title", 3)),
            "answer_string_q2": paired_stats(counts_of(pair, "answer_string", 2)),
        }
    return result


def decide(call_counts, test_counts, paired):
    problems = []
    for ds in SUPPORT_DATASETS:
        n = paired[ds]["paired"]
        if test_counts[ds] != FLASHRAG_COUNTS[ds]:
            problems.append(f"{ds}: test parquet count differs from frozen P0")
        if any(counts[ds]["dump_rows"] != test_counts[ds] for counts in call_counts.values()):
            problems.append(f"{ds}: one or more dumps are incomplete")
        if n < 0.95 * test_counts[ds]:
            problems.append(f"{ds}: too few parsed and paired trajectories")
        if paired[ds]["all_titles_q2"]["n"] < 0.90 * n:
            problems.append(f"{ds}: below 90% matched support annotations")
    primary = macro_stats({ds: data["all_titles_q2"] for ds, data in paired.items()})
    sentences = macro_stats({ds: data["all_sentences_q2"] for ds, data in paired.items()})
    bridges = {ds: paired[ds]["bridge_joint_q1_miss_q3"] for ds in SUPPORT_DATASETS}
    bridge = macro_stats(bridges) if min(v["n"] for v in bridges.values()) >= 100 else None
    checks = {"primary": primary, "secondary_exact_sentences": sentences,
              "exploratory_joint_bridge_miss": bridge}
    if problems or not primary:
        return {"verdict": "INVALID_INPUT", "problems": problems, **checks}
    deltas = [paired[ds]["all_titles_q2"]["delta"] for ds in SUPPORT_DATASETS]
    priority = None
    if primary["delta"] <= -0.02 and primary["ci95"][1] < 0 and max(deltas) < 0:
        verdict = "GO_PREFIX_REPLAY"
        priority = ("bridge_recovery" if bridge and bridge["delta"] <= -0.02 and bridge["ci95"][1] < 0
                    and max(v["delta"] for v in bridges.values()) < 0 else "localize_first_query_or_stopping")
    elif primary["ci95"][0] > -0.02:
        verdict = ("REVIEW_TITLE_SENTENCE_DISAGREEMENT"
                   if sentences and sentences["delta"] <= -0.02 and sentences["ci95"][1] < 0
                   else "NO_MATERIAL_SUPPORT_LOSS")
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "problems": [], "replay_priority": priority, **checks}


def revision(directory: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def atomic_json(path: Path, payload: dict) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=ROOT / "research/data/evidence_dev_v1.json.gz")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        reference = json.load(stream)
    if reference["schema_version"] != 1:
        parser.error("unsupported reference schema")
    print("Reading P0 test parquet and matching public annotations", file=sys.stderr, flush=True)
    tests, test_counts = test_questions(args.test_parquet)
    references = {ds: {key: ref for key, ref in reference["datasets"][ds]["items"].items()
                       if key in tests[ds] and set(ref["answers"]) & answer_keys(tests[ds][key])}
                  for ds in SUPPORT_DATASETS}
    records, counts, files = {}, {}, {}
    for iteration in ("1", "2", "3"):
        print(f"Reading P0 evaluation iteration {iteration}", file=sys.stderr, flush=True)
        path = resolve_dump(args.eval_root / f"iter{iteration}")
        records[iteration], counts[iteration] = load_run(path, tests, references)
        files[iteration] = {"name": path.name, "sha256": sha256(path)}
    paired = {f"{i}->3": compare_runs(records[i], records["3"]) for i in ("1", "2")}
    decision = decide({i: counts[i] for i in ("1", "3")}, test_counts, paired["1->3"])
    provenance = {"repo_head": revision(ROOT),
                  "drzero_head": revision(args.test_parquet.resolve().parent.parent),
                  "protocol_sha256": sha256(ROOT / PROTOCOL),
                  "test_sha256": sha256(args.test_parquet), "dumps": files,
                  "references_sha256": sha256(args.references)}
    by_run = {i: {ds: summarize(list(rows.values())) for ds, rows in datasets.items()}
              for i, datasets in records.items()}

    compact = {
        "protocol": PROTOCOL, "decision": decision, "provenance": provenance,
        "input_counts": {ds: {"test": test_counts[ds], "reference_match": len(references.get(ds, {})),
                              "eval": {i: {**counts[i][ds], "parsed": len(records[i][ds])}
                                       for i in records}} for ds in MULTI},
        "iter1_to_3": paired["1->3"], "iter2_to_3": paired["2->3"],
        "focal_runs": {i: {ds: {k: data[ds][k] for k in
                               ("queries_per_question", "reached_query", "support_titles", "all_titles")}
                           for ds in SUPPORT_DATASETS} for i, data in by_run.items()},
    }
    detailed = {**compact, "by_run": by_run}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "retrieval_diag.json", detailed)
    atomic_json(args.out_dir / "retrieval_diag_return.json", compact)
    print(f"{decision['verdict']}: {args.out_dir / 'retrieval_diag_return.json'}")
    return 2 if decision["verdict"] == "INVALID_INPUT" else 0


if __name__ == "__main__":
    sys.exit(main())
