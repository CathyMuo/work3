"""Audit genuine Wikipedia evidence in the frozen P0 iteration-3 records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from work3.evidence_ref import answer_keys, contains, question_key, sha256
from work3.p0_retrieval_diag import (
    ROOT, SUPPORT_DATASETS, TraceError, dataset_name, question_of, resolve_dump,
    revision, test_questions, trace_of,
)

SEED = 20261122
CAPS = {"missing_with_source": 60, "evidence_present": 30}
AUDIT_V1 = "rsi_wikipedia_input_audit_v1"
AUDIT_W0C = "rsi_wikipedia_source_cache_audit_v1"
CURRENT = 3


def support_facts(reference: dict) -> set[tuple[str, str]]:
    return {(title, sentence) for title, sentences in reference["titles"].items()
            for sentence in sentences}


def source_dumps(root: Path | None) -> dict[int, Path]:
    return {number: resolve_dump(root / f"iter{number}") for number in range(CURRENT)} if root else {}


def source_provenance(caches: dict[int, Path]) -> dict:
    return {str(number): {"dump_name": path.name, "sha256": sha256(path)}
            for number, path in sorted(caches.items())}


def earlier_matches(path, remaining):
    best = {ds: {} for ds in SUPPORT_DATASETS}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            ds = dataset_name(item["data_source"])
            if ds not in remaining or not remaining[ds]:
                continue
            try:
                trace = trace_of(item["output"])
            except TraceError:
                continue
            if trace.errors:
                continue
            for query, group in enumerate(trace.queries, 1):
                for rank, (title, body) in enumerate(group, 1):
                    for sentence in remaining[ds].get(title, ()):
                        if contains(body, sentence):
                            fact = title, sentence
                            candidate = len(body), (line_no, query, rank)
                            if fact not in best[ds] or candidate < best[ds][fact]:
                                best[ds][fact] = candidate
    return best


def find_sources(passages, proposals, seen, caches, on_scan):
    found = {ds: {} for ds in SUPPORT_DATASETS}
    remaining = {ds: defaultdict(set) for ds in SUPPORT_DATASETS}
    for ds in SUPPORT_DATASETS:
        for key, (_, _, missing) in proposals[ds].items():
            if seen[ds][key] == 1:
                for title, sentence in missing:
                    remaining[ds][title].add(sentence)
        for title, sentences in list(remaining[ds].items()):
            for sentence in list(sentences):
                matches = [(len(body), locator) for body, locator in passages[ds][title].items()
                           if contains(body, sentence)]
                if matches:
                    found[ds][title, sentence] = (CURRENT, *min(matches)[1])
                    sentences.remove(sentence)
            if not sentences:
                del remaining[ds][title]
    if on_scan:
        on_scan(1, len(caches) + 1)
    for finished, number in enumerate(sorted(caches, reverse=True), 2):
        for ds, matches in earlier_matches(caches[number], remaining).items():
            for fact, (_, locator) in matches.items():
                found[ds][fact] = (number, *locator)
                title, sentence = fact
                remaining[ds][title].remove(sentence)
                if not remaining[ds][title]:
                    del remaining[ds][title]
        if on_scan:
            on_scan(finished, len(caches) + 1)
    return found


def collect(dump: Path, tests: dict, refs: dict, caches=None, on_scan=None) -> tuple[dict, list[dict]]:
    caches = caches or {}
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    seen = {ds: Counter() for ds in SUPPORT_DATASETS}
    proposals = {ds: {} for ds in SUPPORT_DATASETS}
    passages = {ds: defaultdict(dict) for ds in SUPPORT_DATASETS}

    with dump.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            ds = dataset_name(item["data_source"])
            if ds not in counts:
                continue
            counter = counts[ds]
            counter["dump_rows"] += 1
            key = question_key(question_of(item["input"]))
            seen[ds][key] += 1
            try:
                trace = trace_of(item["output"])
            except TraceError as error:
                counter[f"trace_rejected:{error}"] += 1
                continue
            if trace.errors:
                counter["tool_errors"] += 1
                continue
            counter["trace_parsed_without_tool_error"] += 1
            for query, group in enumerate(trace.queries, 1):
                for rank, (title, body) in enumerate(group, 1):
                    if body:
                        passages[ds][title].setdefault(body, (line_no, query, rank))
            if key not in tests[ds]:
                counter["unmatched_test_question"] += 1
                continue
            if key not in refs[ds]:
                counter["no_matching_support_annotation"] += 1
                continue
            facts = support_facts(refs[ds][key])
            if len({title for title, _ in facts}) < 2 or any(not sent for _, sent in facts):
                counter["incomplete_support_sentences"] += 1
                continue
            counter["complete_support_and_valid_trace"] += 1
            score = float(item["score"])
            if score not in (0, 1):
                raise ValueError(f"Non-binary P0 score at line {line_no}")
            if score == 1:
                counter["originally_correct"] += 1
                continue
            counter["originally_wrong"] += 1
            if not trace.queries:
                counter["wrong_without_any_logged_search"] += 1
                continue
            visible = defaultdict(set)
            for group in trace.queries:
                for title, body in group:
                    visible[title].add(body)
            missing = sorted((title, sentence) for title, sentence in facts
                             if not any(contains(body, sentence) for body in visible[title]))
            proposals[ds][key] = (line_no, len(trace.queries), missing)

    found = find_sources(passages, proposals, seen, caches, on_scan)
    selected = []
    for ds in SUPPORT_DATASETS:
        counter = counts[ds]
        counter["duplicate_question_rows"] = sum(n for n in seen[ds].values() if n > 1)
        cohorts = defaultdict(list)
        for key, (line_no, query_count, missing) in proposals[ds].items():
            if seen[ds][key] != 1:
                continue
            locators = [found[ds][fact] for fact in missing if fact in found[ds]]
            if not missing:
                cohort = "evidence_present"
            elif len(locators) == len(missing):
                cohort = "missing_with_source"
            else:
                counter["missing_without_all_source_passages"] += 1
                if caches:
                    counter["unresolved_partial_sources" if locators else "unresolved_no_source"] += 1
                continue
            counter[cohort] += 1
            candidate = {
                "dataset": ds, "question_key": key, "cohort": cohort, "dump_line": line_no,
                "logged_queries": query_count, "missing_sentences": len(missing),
                "support_passages": sorted(set(locators if caches else (loc[1:] for loc in locators))),
            }
            if caches and locators:
                floor = min(locator[0] for locator in locators)
                counter[f"missing_with_source_floor_iter{floor}"] += 1
                candidate["earliest_source_iteration"] = floor
            cohorts[cohort].append(candidate)
        for cohort, cap in CAPS.items():
            cases = sorted(cohorts[cohort], key=lambda item: hashlib.sha256(
                f"{SEED}:{ds}:{item['question_key']}".encode()).hexdigest())
            selected.extend(cases[:cap])
            counter[f"selected_{cohort}"] = min(len(cases), cap)
            if caches and cohort == "missing_with_source":
                for item in cases[:cap]:
                    floor = item["earliest_source_iteration"]
                    counter[f"selected_missing_with_source_floor_iter{floor}"] += 1
    return {ds: dict(sorted(items.items())) for ds, items in counts.items()}, selected


def validate_baseline(summary, baseline):
    if baseline["protocol"] != AUDIT_V1 or baseline["seed"] != SEED:
        raise ValueError("Expected the original same-round W0b audit summary")
    for key in ("dump_sha256", "test_sha256", "references_sha256"):
        if summary["provenance"][key] != baseline["provenance"][key]:
            raise ValueError(f"Original W0b summary does not match input {key}")
    for ds in SUPPORT_DATASETS:
        row = summary["by_dataset"][ds]
        without_expansion = {key: value for key, value in row.items()
                             if "_floor_iter" not in key and not key.startswith("unresolved_")}
        pool = row.get("missing_with_source_floor_iter3", 0)
        added = row.get("missing_with_source", 0) - pool
        without_expansion["missing_without_all_source_passages"] = (
            row.get("missing_without_all_source_passages", 0) + added)
        without_expansion["missing_with_source"] = pool
        without_expansion["selected_missing_with_source"] = min(pool, CAPS["missing_with_source"])
        old = baseline["by_dataset"][ds]
        if any(without_expansion.get(key, 0) != old.get(key, 0)
               for key in without_expansion.keys() | old.keys()):
            raise ValueError(f"Same-round audit counts changed since W0b for {ds}")


def main(argv=None, *, on_scan=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-root", type=Path)
    evaluation.add_argument("--eval-dump", type=Path, help="explicit original iteration-3 merged JSONL")
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=ROOT / "research/data/evidence_dev_v1.json.gz")
    parser.add_argument("--source-cache-root", type=Path, help="original P0 eval tree with iter0 through iter2")
    parser.add_argument("--baseline-summary", type=Path, help="same-round W0b input summary")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error("Use a new output directory")
    if bool(args.source_cache_root) != bool(args.baseline_summary):
        parser.error("Cache expansion requires both --source-cache-root and --baseline-summary")
    dump = resolve_dump(args.eval_dump or args.eval_root / "iter3")
    caches = source_dumps(args.source_cache_root)
    provenance = {
        "repo_revision": revision(ROOT), "drzero_revision": revision(args.test_parquet.resolve().parent.parent),
        "dump_name": dump.name, "dump_sha256": sha256(dump), "test_sha256": sha256(args.test_parquet),
        "references_sha256": sha256(args.references),
    }
    if caches:
        provenance["source_cache_dumps"] = source_provenance(caches)
        cache_hashes = {file["sha256"] for file in provenance["source_cache_dumps"].values()}
        if len(cache_hashes | {provenance["dump_sha256"]}) != len(caches) + 1:
            raise ValueError("Source cache iterations must refer to distinct evaluation dumps")
    tests, totals = test_questions(args.test_parquet)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        ref = json.load(stream)
    if ref["schema_version"] != 1:
        raise ValueError("Unsupported evidence annotation schema")
    references = {
        ds: {key: item for key, item in ref["datasets"][ds]["items"].items()
             if key in tests[ds] and set(item["answers"]) & answer_keys(tests[ds][key])}
        for ds in SUPPORT_DATASETS
    }
    counts, selected = collect(dump, tests, references, caches, on_scan)
    summary = {
        "protocol": AUDIT_W0C if caches else AUDIT_V1, "seed": SEED,
        "model_source": "P0 iteration-3 saved evaluations",
        "provenance": provenance,
        "by_dataset": {ds: {"test_rows": totals[ds], "matched_references": len(references[ds]),
                            **counts[ds]} for ds in SUPPORT_DATASETS},
        "ready_for_replay_preparation": all(
            counts[ds].get("selected_missing_with_source", 0) == CAPS["missing_with_source"]
            for ds in SUPPORT_DATASETS),
    }
    if caches:
        validate_baseline(summary, json.loads(args.baseline_summary.read_text(encoding="utf-8")))
        provenance["baseline_summary_sha256"] = sha256(args.baseline_summary)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.out_dir / "wiki_rsi_manifest.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in selected), encoding="utf-8")
    if caches:
        provenance["manifest_sha256"] = sha256(manifest)
    (args.out_dir / "wiki_rsi_input_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
