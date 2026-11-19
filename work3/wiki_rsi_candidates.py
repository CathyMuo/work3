"""Screen visible one-query actions on Wikipedia states outside the W0c sample."""

import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict

from work3 import p1_agent as solver
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import answer_keys, contains, question_key, sha256, text_key, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, TraceError, dataset_name, question_of, revision, trace_of
from work3.wiki_rsi_audit import support_facts
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w3a_visible_candidates_v1"
LIMIT = 80
MAX_TITLE_ACTIONS = 3
SEED = 20261129
SINGLE_STOP = frozenset("about after again also among before being between during first found from have into "
                        "last later made many more most other over part same some such than that their them then "
                        "these they this those through time under upon used very what when where which while "
                        "with would years".split())


def phrases(text):
    words = text_key(text).split()
    found = set()
    for start, word in enumerate(words):
        if len(word) >= 4 and not word.isdecimal() and word not in SINGLE_STOP:
            found.add(word)
        for end in range(start + 2, min(len(words), start + 7) + 1):
            found.add(" ".join(words[start:end]))
    return found


def visible_catalog(question, visible, catalog):
    frequency = Counter()
    question_hits = phrases(question)
    for _, body in visible:
        frequency.update(phrases(body) & catalog.keys())
    titles = defaultdict(tuple)
    for cue in (set(frequency) | (question_hits & catalog.keys())):
        for title in catalog[cue]:
            priority = (-frequency[cue], -len(cue.split()), -int(cue in question_hits), title)
            if title not in titles or priority < titles[title]:
                titles[title] = priority
    seen = {name for name, _ in visible if name}
    for name in seen:
        if name not in titles:
            titles[name] = (0, -len(name.split()), 0, name)
    return sorted(titles, key=lambda title: (title in seen, titles[title]))


def action_list(question, titles):
    selected = [title for title in titles if text_key(title) != text_key(question)][:MAX_TITLE_ACTIONS]
    return selected + [question]


def scan_corpus(path, prepared_states, progress, *, missing_by_title=None):
    wanted = set()
    for case in prepared_states:
        wanted.update(phrases(case["question"]))
        for _, body in case["visible"]:
            wanted.update(phrases(body))
    catalog = defaultdict(set)
    digest = hashlib.sha256()
    detail = "scanning_title_catalog" if missing_by_title is None else "scanning_candidate_evidence"
    missing_by_title = missing_by_title or {}
    title_rows = Counter()
    found = set()
    progress.update("running", detail=detail)
    size = path.stat()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            article = json.loads(line)
            name, _, body = article["contents"].partition("\n")
            title = title_key(name)
            cue = text_key(title)
            if cue in wanted:
                catalog[cue].add(title)
            if title in missing_by_title:
                title_rows[title] += 1
                normalized = text_key(body)
                found.update((title, sentence) for sentence in missing_by_title[title]
                             if contains(normalized, sentence))
            rows += 1
            if rows % 100000 == 0:
                progress.update("running", detail=detail, metrics={"corpus_rows": rows})
    end = path.stat()
    if (size.st_size, size.st_mtime_ns) != (end.st_size, end.st_mtime_ns) or not rows:
        raise ValueError("Wikipedia corpus was empty or changed during the title scan")
    return catalog, digest.hexdigest(), rows, title_rows, found


def build_pool(dump, tests, refs, blocked_pages):
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    seen = Counter()
    pool = {ds: [] for ds in SUPPORT_DATASETS}
    for _, row in w1.read_lines(dump):
        ds = dataset_name(row["data_source"])
        if ds not in pool:
            continue
        stats = counts[ds]
        stats["original_eval_rows"] += 1
        qid = question_key(question_of(row["input"]))
        seen[ds, qid] += 1
        score = float(row["score"])
        if score not in (0, 1):
            raise ValueError("Expected binary P0 evaluation scores")
        if score != 0:
            continue
        stats["original_wrong"] += 1
        aliases = tests[ds].get(qid)
        ref = refs[ds].get(qid)
        if not aliases or not ref or not (set(ref["answers"]) & answer_keys(aliases)):
            stats["unmatched_reference"] += 1
            continue
        facts = support_facts(ref)
        pages = set(ref["titles"])
        if len(pages) < 2 or any(not sentence for _, sentence in facts):
            stats["incomplete_reference"] += 1
            continue
        if pages & blocked_pages:
            stats["overlaps_diagnostic_pages"] += 1
            continue
        try:
            trace = trace_of(row["output"])
        except TraceError:
            stats["invalid_original_trace"] += 1
            continue
        if trace.errors or not trace.queries:
            stats["no_successful_original_search"] += 1
            continue
        visible = [doc for group in trace.queries for doc in group]
        gap = sorted((title, sentence) for title, sentence in facts
                     if not any(key == title and contains(body, sentence) for key, body in visible))
        if not gap:
            stats["already_has_support"] += 1
            continue
        pool[ds].append((qid, row, pages, gap, visible))
    for ds, rows in pool.items():
        pool[ds] = sorted((item for item in rows if seen[ds, item[0]] == 1),
                          key=lambda item: hashlib.sha256(f"{SEED}:{ds}:{item[0]}".encode()).digest())
        counts[ds]["unique_pool"] = len(pool[ds])
    return pool, counts


def choose_states(tok, test, pool, counts, blocked_pages, *, limit=LIMIT):
    wanted = {(ds, row[0]) for ds, records in pool.items() for row in records}
    initial = w1.initial_messages(test, wanted)
    used = set(blocked_pages)
    states = []
    for ds, records in pool.items():
        stats = counts[ds]
        for qid, row, pages, gap, visible in records:
            if stats["selected"] == limit:
                break
            if pages & used:
                stats["overlaps_selected_pages"] += 1
                continue
            conversation = initial.get((ds, qid))
            if not conversation:
                stats["missing_original_prompt"] += 1
                continue
            question = question_of(conversation[-1]["content"])
            if question_key(question) != qid:
                raise ValueError("Original question identity changed in candidate screening")
            try:
                prefix, _, _, _ = w1.reconstruct(tok, conversation, row)
            except w1.Excluded as error:
                stats[f"history_excluded:{error}"] += 1
                continue
            if sum(message["role"] == "assistant" for message in prefix) + 2 > solver.MAX_TURNS:
                stats["excluded_turn_budget"] += 1
                continue
            totals = Counter(title for title, _ in gap)
            states.append({"dataset": ds, "qid": qid, "question": question, "visible": visible,
                           "missing_facts": gap, "support_titles": sorted(pages),
                           "probe_title": min(totals, key=lambda title: (-totals[title], title))})
            stats["selected"] += 1
            used.update(pages)
    return states


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    w2a_prep = w1.read_json(run_dir / "wiki_rsi_w2a_prepare_summary.json")
    w2a = w1.read_json(run_dir / "w2a_prepared.json")
    service = w1.read_json(run_dir / "wiki_rsi_w2a_retriever_check.json")
    answer = w1.read_json(run_dir / "wiki_rsi_w2_answer_summary.json")
    if (sha256(run_dir / "w2a_prepared.json") != w2a_prep["w2a_prepared_sha256"]
            or not service["passed"] or answer["completion_jobs"] < 1):
        raise ValueError("Use the completed original W2 run")
    dump, test = w1.input_files(args)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    provenance = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
                  "references_sha256": sha256(args.references), "manifest_sha256": sha256(manifest)}
    if any(w2a_prep["inputs"].get(key) != digest for key, digest in provenance.items()):
        raise ValueError("Use the original P0 and W0c inputs")
    preparation = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, preparation)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tests, _ = w1.test_questions(test)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = {ds: section["items"] for ds, section in json.load(stream)["datasets"].items()}
    blocked_pages = {title for _, row in w1.read_lines(manifest)
                     for title in refs[row["dataset"]][row["question_key"]]["titles"]}
    pool, counts = build_pool(dump, tests, refs, blocked_pages)
    cases = choose_states(tok, test, pool, counts, blocked_pages)
    if not all(counts[ds]["selected"] for ds in SUPPORT_DATASETS):
        raise ValueError("One dataset has no page-disjoint terminal states")
    path = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    corpus, digest, corpus_rows, _, _ = scan_corpus(path, cases, progress)
    result_path = run_dir / "w3_candidates_prepared.json"
    for case in cases:
        titles = visible_catalog(case["question"], case.pop("visible"), corpus)
        case["visible_title_count"] = len(titles)
        case["candidates"] = action_list(case["question"], titles)
        case["target_in_catalog"] = case["probe_title"] in titles
        case["target_in_candidates"] = case["probe_title"] in case["candidates"]
    atomic_json(result_path, {"protocol": PROTOCOL, "records": cases, "anchors": w2a["anchors"],
                             "retriever_id": service["retriever_id"]})
    report = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "by_dataset": counts,
              "inputs": {**provenance, "corpus_sha256": digest, "corpus_rows": corpus_rows},
              "excluded_diagnostic_support_pages": len(blocked_pages),
              "prepared_sha256": sha256(result_path)}
    atomic_json(run_dir / "wiki_rsi_w3_candidates_prepare_summary.json", report)
    progress.update("done", len(cases), len(cases), detail="ready",
                    metrics={ds: counts[ds]["selected"] for ds in SUPPORT_DATASETS})
    print(json.dumps(report, indent=2), flush=True)
    return 0


def best_of_pool(measures):
    return {metric: any(row[metric] for row in measures) for metric in ("any_missing_fact", "all_missing_facts")}


def summarize(records):
    result = {}
    for ds in SUPPORT_DATASETS:
        rows = [row for row in records if row["dataset"] == ds]
        pairs = [{key: {"correct": int(row[key]["all_missing_facts"])} for key in ("top1", "original_question")}
                 for row in rows]
        result[ds] = {
            "n": len(rows),
            "target_in_visible_catalog": sum(row["target_in_catalog"] for row in rows),
            "target_in_capped_candidates": sum(row["target_in_candidates"] for row in rows),
            "states_with_title_candidates": sum(row["candidate_count"] > 1 for row in rows),
            "candidate_slots_used": sum(row["candidate_count"] for row in rows),
            "coverage": {arm: {metric: sum(row[arm][metric] for row in rows)
                               for metric in ("any_missing_fact", "all_missing_facts")}
                         for arm in ("top1", "original_question", "best_single_candidate", "title_probe")},
            "top1_vs_original_all_missing": w1.paired(pairs, "top1", "original_question"),
        }
    return result


def run(args, run_dir, progress):
    start = time.monotonic()
    path = run_dir / "w3_candidates_prepared.json"
    report_path = run_dir / "wiki_rsi_w3_candidates_prepare_summary.json"
    report, data = w1.read_json(report_path), w1.read_json(path)
    if (report["protocol"] != PROTOCOL or data["protocol"] != PROTOCOL
            or report["prepared_sha256"] != sha256(path) or data["retriever_id"] != args.retriever_id):
        raise ValueError("Candidate check must reuse the prepared split and original service")
    cases = data["records"]
    progress.update("running", 0, len(cases), detail="checking_retriever")
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    parity = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w3_candidates_retriever_check.json", parity)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(parity, indent=2), flush=True)
        return 2
    records, queries = [], 0
    for offset in range(0, len(cases), 6):
        batch = cases[offset:offset + 6]
        requested = list(dict.fromkeys(query for case in batch
                                       for query in case["candidates"] + [case["probe_title"]]))
        returned = dict(zip(requested, retriever.search(requested), strict=True))
        queries += len(requested)
        for case in batch:
            arms = [{**w2.measure(returned[action], case["missing_facts"]),
                     "query": action, "passages": returned[action]} for action in case["candidates"]]
            probe = {**w2.measure(returned[case["probe_title"]], case["missing_facts"]),
                     "query": case["probe_title"], "passages": returned[case["probe_title"]]}
            records.append({key: case[key] for key in ("dataset", "qid", "target_in_catalog",
                                                       "target_in_candidates")} | {
                "candidate_count": len(arms), "candidate_results": arms, "top1": arms[0],
                "original_question": arms[-1], "best_single_candidate": best_of_pool(arms), "title_probe": probe})
        progress.update("running", len(records), len(cases), detail="querying_wikipedia",
                        dataset=batch[-1]["dataset"], metrics={"queries": queries})
    atomic_json(run_dir / "w3_candidates_cases.json", {"protocol": PROTOCOL, "records": records})
    elapsed = round(time.monotonic() - start, 3)
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": parity,
               "preparation_sha256": sha256(report_path), "by_dataset": summarize(records),
               "physical_queries_excluding_anchors": queries, "wall_seconds": elapsed}
    atomic_json(run_dir / "wiki_rsi_w3_candidates_summary.json", summary)
    progress.update("done", len(records), len(cases), detail="completed",
                    metrics={"queries": queries, "wall_seconds": elapsed})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
