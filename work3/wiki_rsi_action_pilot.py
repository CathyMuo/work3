"""Prepare independent execution-verified action pairs and a natural QA evaluation."""

import gzip
import json
from collections import Counter, defaultdict

from work3 import wiki_rsi_candidates as w3
from work3 import wiki_rsi_natural as w5a
from work3 import wiki_rsi_natural_query as w5b
from work3 import wiki_rsi_repair_answer as w4c
from work3 import wiki_rsi_repair_audit as w4b
from work3 import wiki_rsi_repair_pool as pool
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w6a_independent_action_pool_v1"
W5B_SUMMARY = "wiki_rsi_w5_natural_query_summary.json"
PREPARED = "w6_action_pilot_prepared.json"
PREP_SUMMARY = "wiki_rsi_w6_action_pilot_prepare_summary.json"
SOURCE_CAP = 2048
TRAIN_CAP, MIN_TRAIN = 160, 80
EVAL_CAPS = {"not_literal": 64, "literal": 128}
EVAL_MINS = {"not_literal": 32, "literal": 80}
MIN_EVAL_WITNESSES = 10


def literal_title(question, visible, title):
    phrase = text_key(title)
    return any(contains(text_key(text), phrase)
               for text in [question, *(field for passage in visible for field in passage)])


def blocked_history(run_dir, refs):
    older = w1.read_json(run_dir / w5a.W3_CASES)
    recent = w1.read_json(run_dir / w5a.PREPARED)
    films = w1.read_json(run_dir / w4c.AUDIT_CASES)
    eval_blocked = {title for batch in (older, recent) for row in batch["records"]
                    for title in row["support_titles"]}
    eval_blocked.update(title for _, row in w1.read_lines(run_dir / "w0/wiki_rsi_manifest.jsonl")
                        for title in refs[row["dataset"]][row["question_key"]]["titles"])
    film_pages = {title_key(row[key]) for row in films["selected"] for key in ("source", "target", "foil")}
    eval_blocked.update(film_pages)
    train_blocked = {title_key(title) for dataset in refs.values()
                     for item in dataset.values() for title in item["titles"]}
    train_blocked.update(eval_blocked)
    train_blocked.update(title_key(row["source"]) for row in films["source_traces"])
    train_blocked.update(title_key(name) for name in films["person_returns"])
    return eval_blocked, train_blocked


def choose_evaluation(tok, test, rows, blocked):
    buckets = {key: {ds: [] for ds in SUPPORT_DATASETS} for key in EVAL_CAPS}
    for dataset, members in rows.items():
        for record in members:
            _, row, _, gap, visible = record
            question = w3.question_of(row["input"])
            label = "literal" if literal_title(question, visible, gap[0][0]) else "not_literal"
            buckets[label][dataset].append(record)
    records, reports = [], {}
    used = set(blocked)
    for label, limit in EVAL_CAPS.items():
        counts = {ds: Counter(eligible=len(buckets[label][ds])) for ds in SUPPORT_DATASETS}
        selection = w3.choose_states(tok, test, buckets[label], counts, used, limit=limit)
        for case in selection:
            if literal_title(case["question"], case["visible"], case["probe_title"]) != (label == "literal"):
                raise ValueError("Title stratum changed after reconstructing the original question")
            case["title_stratum"] = label
            used.update(case["support_titles"])
        records.extend(selection)
        reports[label] = counts
    return records, reports


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    prior = w1.read_json(run_dir / w5a.PREP_SUMMARY)
    baseline = w1.read_json(run_dir / w5a.PREPARED)
    w5b_prep = w1.read_json(run_dir / w5b.PREP_SUMMARY)
    result = w1.read_json(run_dir / W5B_SUMMARY)
    w4_prep = w1.read_json(run_dir / w4c.PREP_SUMMARY)
    if (prior["protocol"] != w5a.PROTOCOL or baseline["protocol"] != w5a.PROTOCOL
            or result["protocol"] != w5b.PROTOCOL or w5b_prep["protocol"] != w5b.PROTOCOL
            or result["decision"] != "qualifies_for_independent_action_training_pilot"
            or result["preparation_sha256"] != sha256(run_dir / w5b.PREP_SUMMARY)
            or prior["prepared_sha256"] != sha256(run_dir / w5a.PREPARED)
            or w5b_prep["w5a_prepared_sha256"] != prior["prepared_sha256"]
            or not result["retriever"]["passed"] or result["retriever"]["retriever_id"] != baseline["retriever_id"]
            or prior["inputs"]["w3_prepared_sha256"] != sha256(run_dir / w5a.W3_CASES)
            or w4_prep["w4b_cases_sha256"] != sha256(run_dir / w4c.AUDIT_CASES)):
        raise ValueError("Use the completed qualified W5b run and pre-existing diagnostic splits")
    dump, test = w1.input_files(args)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    inputs = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
              "references_sha256": sha256(args.references), "manifest_sha256": sha256(manifest)}
    if any(prior["inputs"][key] != digest for key, digest in inputs.items()):
        raise ValueError("Use the byte-identical P0 and W0c inputs")
    checkpoint = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, checkpoint)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tests, _ = w1.test_questions(test)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = {ds: section["items"] for ds, section in json.load(stream)["datasets"].items()}
    eval_blocked, train_blocked = blocked_history(run_dir, refs)
    rows, counts = w3.build_pool(dump, tests, refs, eval_blocked)
    cases, stratified = choose_evaluation(tok, test, w5a.select_pool(rows, counts), eval_blocked)
    if not cases:
        raise ValueError("No independent evaluation states")
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    wanted = defaultdict(set)
    for case in cases:
        for title, sentence in case["missing_facts"]:
            wanted[title].add(sentence)
    catalog, digest, size, _, found = w3.scan_corpus(corpus, cases, progress, missing_by_title=wanted)
    expected = prior["inputs"]["corpus_sha256"], prior["inputs"]["corpus_rows"]
    if (digest, size) != expected:
        raise ValueError("Use the original Wikipedia corpus")
    for case in cases:
        case["queries"], case["visible_titles"] = w5a.visible_queries(case["question"], case["visible"], catalog)
        case["strict_corpus_all"] = all(tuple(fact) in found for fact in case["missing_facts"])
    train_blocked.update(title_key(title) for case in cases for title, _ in case["visible"])
    train_digest, train_size, sources = pool.scan_sources(corpus, train_blocked, progress)
    if (train_digest, train_size) != expected:
        raise ValueError("Corpus changed between evaluation and independent source scans")
    train_sources = [sources[key] for key in sorted(sources, key=pool.order_key)[:SOURCE_CAP]]
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "anchors": baseline["anchors"], "retriever_id": baseline["retriever_id"],
                       "evaluation": cases, "train_sources": train_sources, "train_blocked": sorted(train_blocked)})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "prepared_sha256": sha256(path),
               "w5b_summary_sha256": sha256(run_dir / W5B_SUMMARY), "corpus_sha256": digest,
               "corpus_rows": size, "evaluation_pool": counts, "evaluation_strata": stratified,
               "train_source_candidates": len(sources), "train_sources_selected": len(train_sources),
               "excluded_previous_evaluation_support_pages": len(eval_blocked)}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(cases), len(cases), detail="ready")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def select_training(histories, returned, blocked):
    observations = {name: w4b.birth_entries(name, passages) for name, passages in returned.items()}
    used, selected = set(blocked), []
    for trace in histories:
        source, prefix = trace["source"], trace["prefix"]
        pages = {title_key(title) for title, _ in prefix}
        if pages & blocked:
            continue
        _, options = w4b.candidates(source, prefix, trace["people"], returned, observations, "normalized_one_witness")
        for case in options:
            pair = {title_key(case[key]) for key in ("source", "target", "foil")}
            if pair & used:
                continue
            used.update(pair)
            visible = [list(doc) for doc in w2.normalized_docs(prefix)]
            selected.append({**case, "visible": visible, "positive_query": case["target"],
                             "negative_query": case["foil"], "positive_return": returned[case["target"]],
                             "negative_return": returned[case["foil"]]})
            break
        if len(selected) == TRAIN_CAP:
            break
    return selected


def eval_report(rows):
    report = {}
    for ds in SUPPORT_DATASETS:
        report[ds] = {}
        for label in EVAL_CAPS:
            members = [row for row in rows if row["dataset"] == ds and row["title_stratum"] == label]
            report[ds][label] = {"n": len(members), **{
                key: sum(row[key] for row in members)
                for key in (*w5a.FIXED_ARMS, "known_title", "any_visible_candidate", "strict_corpus_all")
            }}
    return report


def readiness(train, report):
    return len(train) >= MIN_TRAIN and all(
        all(row[label]["n"] >= minimum for label, minimum in EVAL_MINS.items())
        and row["literal"]["known_title"] >= MIN_EVAL_WITNESSES
        for row in report.values()
    )


def run(args, run_dir, progress):
    path = run_dir / PREPARED
    data, prep = w1.read_json(path), w1.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or prep["protocol"] != PROTOCOL
            or prep["prepared_sha256"] != sha256(path) or data["retriever_id"] != args.retriever_id
            or prep["w5b_summary_sha256"] != sha256(run_dir / W5B_SUMMARY)):
        raise ValueError("Use the frozen independent pool preparation and original retriever")
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w6_action_pilot_retriever_check.json", check)
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    blocked = set(data["train_blocked"])
    traces, returned = w4b.collect(retriever, data["train_sources"], blocked, progress)
    train = select_training(traces, returned, blocked)
    evaluation, eval_queries = [], 0
    cases = data["evaluation"]
    for offset in range(0, len(cases), 8):
        batch = cases[offset:offset + 8]
        queries = list(dict.fromkeys(query for case in batch for query in w5a.case_queries(case)))
        results = dict(zip(queries, retriever.search(queries), strict=True))
        eval_queries += len(queries)
        for case in batch:
            row = w5a.score_case(case, {query: results[query] for query in w5a.case_queries(case)})
            evaluation.append({**row, "title_stratum": case["title_stratum"]})
        progress.update("running", len(evaluation), len(cases), detail="querying_wikipedia",
                        metrics={"training_pairs": len(train), "queries": eval_queries})
    report = eval_report(evaluation)
    train_path = run_dir / "w6_action_train_pairs.json"
    eval_path = run_dir / "w6_action_eval_baselines.json"
    atomic_json(train_path, {"protocol": PROTOCOL, "records": train})
    atomic_json(eval_path, {"protocol": PROTOCOL, "records": evaluation})
    ready = readiness(train, report)
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": check,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY),
               "train_pairs_sha256": sha256(train_path), "evaluation_baselines_sha256": sha256(eval_path),
               "train_pairs": len(train), "train_roles": Counter(row["role"] for row in train),
               "train_sources_examined": len(traces), "train_person_queries": len(returned),
               "evaluation_by_title_stratum": report, "evaluation_physical_queries": eval_queries,
               "minimums": {"train_pairs": MIN_TRAIN, "evaluation_per_dataset": EVAL_MINS,
                            "visible_known_title_witnesses_per_dataset": MIN_EVAL_WITNESSES},
               "decision": "ready_for_independent_gpu_pilot" if ready else "insufficient_independent_data"}
    atomic_json(run_dir / "wiki_rsi_w6_action_pilot_summary.json", summary)
    progress.update("done", len(evaluation), len(evaluation), detail="completed", metrics={"training_pairs": len(train)})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
