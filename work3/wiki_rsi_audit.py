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


def support_facts(reference: dict) -> set[tuple[str, str]]:
    return {(title, sentence) for title, sentences in reference["titles"].items()
            for sentence in sentences}


def collect(dump: Path, tests: dict, refs: dict) -> tuple[dict, list[dict]]:
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

    selected = []
    for ds in SUPPORT_DATASETS:
        counter = counts[ds]
        counter["duplicate_question_rows"] = sum(n for n in seen[ds].values() if n > 1)
        cohorts = defaultdict(list)
        for key, (line_no, query_count, missing) in proposals[ds].items():
            if seen[ds][key] != 1:
                continue
            locators = []
            for title, sentence in missing:
                matches = [(len(body), locator) for body, locator in passages[ds][title].items()
                           if contains(body, sentence)]
                if matches:
                    locators.append(min(matches)[1])
            if not missing:
                cohort = "evidence_present"
            elif len(locators) == len(missing):
                cohort = "missing_with_source"
            else:
                counter["missing_without_all_source_passages"] += 1
                continue
            counter[cohort] += 1
            cohorts[cohort].append({
                "dataset": ds, "question_key": key, "cohort": cohort, "dump_line": line_no,
                "logged_queries": query_count, "missing_sentences": len(missing),
                "support_passages": sorted(set(locators)),
            })
        for cohort, cap in CAPS.items():
            cases = sorted(cohorts[cohort], key=lambda item: hashlib.sha256(
                f"{SEED}:{ds}:{item['question_key']}".encode()).hexdigest())
            selected.extend(cases[:cap])
            counter[f"selected_{cohort}"] = min(len(cases), cap)
    return {ds: dict(sorted(items.items())) for ds, items in counts.items()}, selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    evaluation = parser.add_mutually_exclusive_group(required=True)
    evaluation.add_argument("--eval-root", type=Path)
    evaluation.add_argument("--eval-dump", type=Path, help="explicit original iteration-3 merged JSONL")
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=ROOT / "research/data/evidence_dev_v1.json.gz")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error("Use a new output directory")
    dump = resolve_dump(args.eval_dump or args.eval_root / "iter3")
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
    counts, selected = collect(dump, tests, references)
    summary = {
        "protocol": "rsi_wikipedia_input_audit_v1", "seed": SEED,
        "model_source": "P0 iteration-3 saved evaluations",
        "provenance": {
            "repo_revision": revision(ROOT), "drzero_revision": revision(args.test_parquet.resolve().parent.parent),
            "dump_name": dump.name, "dump_sha256": sha256(dump), "test_sha256": sha256(args.test_parquet),
            "references_sha256": sha256(args.references),
        },
        "by_dataset": {ds: {"test_rows": totals[ds], "matched_references": len(references[ds]),
                            **counts[ds]} for ds in SUPPORT_DATASETS},
        "ready_for_replay_preparation": all(
            counts[ds].get("selected_missing_with_source", 0) == CAPS["missing_with_source"]
            for ds in SUPPORT_DATASETS),
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "wiki_rsi_manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in selected), encoding="utf-8")
    (args.out_dir / "wiki_rsi_input_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
