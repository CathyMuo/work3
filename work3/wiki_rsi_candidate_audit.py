"""Locate evidence loss between the frozen Wikipedia corpus, candidate list and top-3 returns."""

import json
import time
from collections import Counter, defaultdict

from work3 import wiki_rsi_candidates as w3a
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, question_key, sha256, text_key, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, dataset_name, question_of, revision, trace_of
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w3b_candidate_audit_v1"
PREPARED = "w3_candidate_audit_prepared.json"
PREP_SUMMARY = "wiki_rsi_w3_candidate_audit_prepare_summary.json"
W3A_PREPARED = "w3_candidates_prepared.json"
W3A_PREP_SUMMARY = "wiki_rsi_w3_candidates_prepare_summary.json"
W3A_SUMMARY = "wiki_rsi_w3_candidates_summary.json"
W3A_CASES = "w3_candidates_cases.json"
CAPS = (3, 8)
RANKINGS = ("legacy", "question_rank")
FORMS = ("title", "title_question")
METRICS = ("any_missing_fact", "all_missing_facts")


def question_first(question, titles):
    normalized = text_key(question)
    tokens = set(normalized.split()) - w3a.SINGLE_STOP
    return sorted(titles, key=lambda title: (
        not contains(normalized, text_key(title)),
        -len(tokens.intersection(text_key(title).split())),
    ))


def restore_visible(dump, cases):
    wanted = {(case["dataset"], case["qid"]): case for case in cases}
    if len(cases) != len(wanted):
        raise ValueError("Duplicate W3a cases")
    for _, row in w1.read_lines(dump):
        key = dataset_name(row["data_source"]), question_key(question_of(row["input"]))
        if key not in wanted:
            continue
        case = wanted[key]
        trace = trace_of(row["output"])
        if "visible" in case or float(row["score"]) != 0 or trace.errors or not trace.queries:
            raise ValueError("W3a trace no longer uniquely reconstructs a failed search state")
        case["visible"] = [doc for group in trace.queries for doc in group]
    if any("visible" not in case for case in cases):
        raise ValueError("Missing original W3a search history")


def prepare(args, run_dir, progress):
    old_prep = w1.read_json(run_dir / W3A_PREP_SUMMARY)
    old_summary = w1.read_json(run_dir / W3A_SUMMARY)
    frozen = w1.read_json(run_dir / W3A_PREPARED)
    if (any(data["protocol"] != w3a.PROTOCOL for data in (old_prep, old_summary, frozen))
            or sha256(run_dir / W3A_PREPARED) != old_prep["prepared_sha256"]
            or sha256(run_dir / W3A_PREP_SUMMARY) != old_summary["preparation_sha256"]
            or not old_summary["retriever"]["passed"]
            or old_summary["retriever"]["retriever_id"] != frozen["retriever_id"]):
        raise ValueError("Use the completed W3a screen from this run")
    dump, test = w1.input_files(args)
    inputs = {"dump_sha256": sha256(dump), "test_sha256": sha256(test)}
    for key, digest in inputs.items():
        if old_prep["inputs"][key] != digest:
            raise ValueError(f"W3a input changed: {key}")
    cases = frozen["records"]
    restore_visible(dump, cases)
    missing_by_title = defaultdict(set)
    for case in cases:
        for title, sentence in case["missing_facts"]:
            missing_by_title[title].add(sentence)
    path = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    catalog, digest, rows, title_rows, found = w3a.scan_corpus(
        path, cases, progress, missing_by_title=missing_by_title)
    if digest != old_prep["inputs"]["corpus_sha256"] or rows != old_prep["inputs"]["corpus_rows"]:
        raise ValueError("Use the exact W3a Wikipedia corpus")
    for case in cases:
        question, target = case["question"], case["probe_title"]
        legacy = w3a.visible_catalog(question, case.pop("visible"), catalog)
        target_visible = target in legacy
        if (len(legacy) != case["visible_title_count"] or target_visible != case["target_in_catalog"]
                or w3a.action_list(question, legacy) != case["candidates"]):
            raise ValueError("Reconstructed W3a candidate list differs from the frozen list")
        eligible = [title for title in legacy if text_key(title) != text_key(question)]
        rankings = {"legacy": eligible, "question_rank": question_first(question, eligible)}
        case["lists"] = {rank: titles[:max(CAPS)] for rank, titles in rankings.items()}
        case["ranks"] = {rank: titles.index(target) + 1 if target in titles else None
                         for rank, titles in rankings.items()}
        case["corpus"] = [{"title": title, "title_passages": title_rows[title],
                           "sentence": sentence, "exact_match": (title, sentence) in found}
                          for title, sentence in case["missing_facts"]]
    original_cases = run_dir / W3A_CASES
    data = {"protocol": PROTOCOL, "records": cases, "anchors": frozen["anchors"],
            "retriever_id": frozen["retriever_id"], "w3a_cases_sha256": sha256(original_cases)}
    atomic_json(run_dir / PREPARED, data)
    counts = Counter(case["dataset"] for case in cases)
    report = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "by_dataset": counts,
              "inputs": {**inputs, "corpus_sha256": digest, "corpus_rows": rows,
                         "w3a_prepared_sha256": old_prep["prepared_sha256"],
                         "w3a_summary_sha256": sha256(run_dir / W3A_SUMMARY),
                         "w3a_cases_sha256": data["w3a_cases_sha256"]},
              "prepared_sha256": sha256(run_dir / PREPARED)}
    atomic_json(run_dir / PREP_SUMMARY, report)
    progress.update("done", len(cases), len(cases), detail="ready", metrics=counts)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def query_for(title, question, form):
    return title if form == "title" else f"{title}. {question}"


def requested_queries(case):
    question = case["question"]
    titles = [title for choices in case["lists"].values() for title in choices]
    return [question] + [query_for(title, question, form)
                         for title in [*titles, case["probe_title"]] for form in FORMS]


def score_case(case, returned, old):
    question, target = case["question"], case["probe_title"]
    missing = case["missing_facts"]
    target_facts = [fact for fact in missing if fact[0] == target]
    exact = [row["exact_match"] for row in case["corpus"]]
    target_rows = [row for row in case["corpus"] if row["title"] == target]
    target_exact = [row["exact_match"] for row in target_rows]
    queries = {query: {**w2.measure(returned[query], missing), "passages": returned[query]}
               for query in dict.fromkeys(requested_queries(case))}
    probes = {}
    for form in FORMS:
        query = query_for(target, question, form)
        docs = returned[query]
        target_measure = w2.measure(docs, target_facts)
        probes[form] = {**{metric: queries[query][metric] for metric in METRICS},
                        "target_page_in_top3": any(title_key(title) == target for title, _ in docs),
                        "any_target_fact": target_measure["any_missing_fact"],
                        "all_target_facts": target_measure["all_missing_facts"]}
    arms = {"original_question": queries[question]}
    for rank, titles in case["lists"].items():
        for form in FORMS:
            measures = [queries[query_for(title, question, form)] for title in titles]
            arms[f"{rank}_{form}_top1"] = measures[0] if measures else queries[question]
            for cap in CAPS:
                arms[f"{rank}_{form}_best{cap}"] = w3a.best_of_pool([queries[question], *measures[:cap]])
    replay = {arm: w2.normalized_docs(returned[query]) == w2.normalized_docs(old[arm]["passages"])
              for arm, query in (("original_question", question), ("title_probe", target))}
    return {"dataset": case["dataset"], "qid": case["qid"], "ranks": case["ranks"],
            "target_visible": case["target_in_catalog"], "missing_pages": len({title for title, _ in missing}),
            "corpus_any": any(exact), "corpus_all": all(exact),
            "corpus_has_target_page": any(row["title_passages"] for row in target_rows),
            "corpus_any_target": any(target_exact), "corpus_all_target": all(target_exact),
            "probes": probes, "arms": {arm: {metric: row[metric] for metric in METRICS}
                                     for arm, row in arms.items()},
            "queries": queries, "w3a_exact_ordered": replay}


def counts_for(rows):
    counts = Counter()
    for row in rows:
        counts["n"] += 1
        for key in ("corpus_any", "corpus_all", "corpus_has_target_page", "corpus_any_target",
                    "corpus_all_target", "target_visible"):
            counts[key] += row[key]
        for rank, position in row["ranks"].items():
            for cap in CAPS:
                counts[f"{rank}_target_top{cap}"] += position is not None and position <= cap
        for form, probe in row["probes"].items():
            for metric, value in probe.items():
                counts[f"probe_{form}_{metric}"] += value
            counts[f"probe_{form}_page_without_target_fact"] += (
                row["corpus_any_target"] and probe["target_page_in_top3"] and not probe["any_target_fact"])
            counts[f"probe_{form}_missed_page_with_corpus_fact"] += (
                row["corpus_any_target"] and not probe["target_page_in_top3"])
            position = row["ranks"]["legacy"]
            counts[f"probe_{form}_target_fact_but_title_invisible"] += (
                probe["any_target_fact"] and not row["target_visible"])
            counts[f"probe_{form}_target_fact_but_title_after_legacy_top3"] += (
                probe["any_target_fact"] and position is not None and position > 3)
        for arm, result in row["arms"].items():
            for metric, value in result.items():
                counts[f"{arm}_{metric}"] += value
        for arm, same in row["w3a_exact_ordered"].items():
            counts[f"w3a_reproduced_{arm}"] += same
    return {"n": 0, **counts}


def summarize(rows):
    results = {}
    comparisons = (("question_rank_title_top1", "legacy_title_top1"),
                   ("question_rank_title_question_top1", "question_rank_title_top1"),
                   ("question_rank_title_question_top1", "original_question"))
    for dataset in SUPPORT_DATASETS:
        group = [row for row in rows if row["dataset"] == dataset]
        pairs = [{arm: {"correct": int(result["all_missing_facts"])} for arm, result in row["arms"].items()}
                 for row in group]
        results[dataset] = {
            "all_states": counts_for(group),
            "one_missing_page": counts_for(row for row in group if row["missing_pages"] == 1),
            "multiple_missing_pages": counts_for(row for row in group if row["missing_pages"] > 1),
            "one_query_paired_all_missing": {f"{a}_vs_{b}": w1.paired(pairs, a, b)
                                             for a, b in comparisons},
        }
    return results


def run(args, run_dir, progress):
    start = time.monotonic()
    report = w1.read_json(run_dir / PREP_SUMMARY)
    data = w1.read_json(run_dir / PREPARED)
    if (report["protocol"] != PROTOCOL or data["protocol"] != PROTOCOL
            or report["prepared_sha256"] != sha256(run_dir / PREPARED)
            or data["retriever_id"] != args.retriever_id
            or data["w3a_cases_sha256"] != sha256(run_dir / W3A_CASES)):
        raise ValueError("Use the frozen W3b preparation and original retriever")
    cases = data["records"]
    original = w1.read_json(run_dir / W3A_CASES)
    old = {(row["dataset"], row["qid"]): row for row in original["records"]}
    if (original["protocol"] != w3a.PROTOCOL or len(old) != len(original["records"])
            or old.keys() != {(row["dataset"], row["qid"]) for row in cases}):
        raise ValueError("W3a case results do not match the frozen W3b cases")
    progress.update("running", 0, len(cases), detail="checking_retriever")
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    parity = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w3_candidate_audit_retriever_check.json", parity)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(parity, indent=2), flush=True)
        return 2
    records, queries = [], 0
    for offset in range(0, len(cases), 2):
        batch = cases[offset:offset + 2]
        requested = list(dict.fromkeys(query for case in batch for query in requested_queries(case)))
        returned = dict(zip(requested, retriever.search(requested), strict=True))
        queries += len(requested)
        records.extend(score_case(case, returned, old[case["dataset"], case["qid"]]) for case in batch)
        progress.update("running", len(records), len(cases), detail="querying_wikipedia",
                        dataset=batch[-1]["dataset"], metrics={"queries": queries})
    atomic_json(run_dir / "w3_candidate_audit_cases.json", {"protocol": PROTOCOL, "records": records})
    elapsed = round(time.monotonic() - start, 3)
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": parity,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "by_dataset": summarize(records),
               "physical_queries_excluding_anchors": queries, "wall_seconds": elapsed}
    atomic_json(run_dir / "wiki_rsi_w3_candidate_audit_summary.json", summary)
    progress.update("done", len(records), len(cases), detail="completed",
                    metrics={"queries": queries, "wall_seconds": elapsed})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
