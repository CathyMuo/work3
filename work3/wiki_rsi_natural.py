"""Measure observable one-query repair headroom on page-disjoint natural QA."""

import gzip
import json
import re
from collections import Counter, defaultdict

from work3 import wiki_rsi_candidates as w3
from work3 import wiki_rsi_repair_answer as w4c
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w5a_natural_gate_v1"
LIMIT = 192
MIN_ROWS, MIN_PROBE, MIN_GAP = 100, 20, 10
OUT_OF_DOMAIN = re.compile(r"\b(?:film|films|movie|movies|screenplay|director|directed|born|birth|birthdate)\b")
W3_PREP = "wiki_rsi_w3_candidates_prepare_summary.json"
W3_CASES = "w3_candidates_prepared.json"
W4C_SUMMARY = "wiki_rsi_w4_repair_answer_summary.json"
PREPARED = "w5_natural_prepared.json"
PREP_SUMMARY = "wiki_rsi_w5_natural_prepare_summary.json"
FIXED_ARMS = ("original", "linked", "linked_context")


def select_pool(rows, counts):
    selected = {}
    for dataset, records in rows.items():
        eligible = []
        for qid, row, pages, gap, visible in records:
            if len({title for title, _ in gap}) != 1:
                counts[dataset]["excluded_multiple_missing_pages"] += 1
                continue
            question = text_key(w3.question_of(row["input"]))
            if any(OUT_OF_DOMAIN.search(value) for value in (question, *(text_key(title) for title in pages))):
                counts[dataset]["excluded_film_or_birth"] += 1
                continue
            eligible.append((qid, row, pages, gap, visible))
        selected[dataset] = eligible
        counts[dataset]["eligible_single_page"] = len(eligible)
    return selected


def words(text):
    return list(dict.fromkeys(word for word in text_key(text).split()
                              if len(word) > 2 and word not in w3.SINGLE_STOP))


def visible_queries(question, visible, catalog):
    """Rank index titles by a question-relevant visible passage; never read support labels."""
    q_words = words(question)
    q_set = set(q_words)
    appearances = Counter()
    documents = []
    for _, body in visible:
        terms = set(words(body))
        documents.append(terms)
        appearances.update(q_set & terms)
    context = defaultdict(float)
    for (_, body), terms in zip(visible, documents, strict=True):
        relevance = sum(1 / appearances[word] for word in sorted(q_set & terms))
        for phrase in w3.phrases(body) & catalog.keys():
            for title in catalog[phrase]:
                context[title] = max(context[title], relevance)
    titles = w3.visible_catalog(question, visible, catalog)
    seen = {name for name, _ in visible}
    canonical_question = text_key(question)
    ordered = sorted(titles, key=lambda title: (title in seen, -context[title],
                     not contains(canonical_question, text_key(title))))
    ordered = [title for title in ordered if text_key(title) != canonical_question][:3]
    chosen = ordered[0] if ordered else question
    chosen_words = set(words(chosen))
    keywords = [word for word in q_words if word not in chosen_words][:5]
    expanded = f"{chosen} {' '.join(keywords)}" if keywords and ordered else chosen
    return {"original": question, "linked": chosen, "linked_context": expanded}, ordered


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    prior = w1.read_json(run_dir / W3_PREP)
    old = w1.read_json(run_dir / W3_CASES)
    replay = w1.read_json(run_dir / W4C_SUMMARY)
    if (prior["protocol"] != w3.PROTOCOL or old["protocol"] != w3.PROTOCOL
            or prior["prepared_sha256"] != sha256(run_dir / W3_CASES)
            or replay["protocol"] != w4c.PROTOCOL
            or replay["preparation_sha256"] != sha256(run_dir / w4c.PREP_SUMMARY)
            or replay["completion_jobs"] != replay["all_cases"]["n"] * len(w4c.ARMS)):
        raise ValueError("Use the original W3a preparation and the completed W4c replay")
    dump, test = w1.input_files(args)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    inputs = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
              "references_sha256": sha256(args.references), "manifest_sha256": sha256(manifest)}
    if any(prior["inputs"][key] != digest for key, digest in inputs.items()):
        raise ValueError("Use the original P0 and W0c inputs")
    checkpoint = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, checkpoint)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tests, _ = w1.test_questions(test)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = {dataset: section["items"] for dataset, section in json.load(stream)["datasets"].items()}
    blocked = {title for row in old["records"] for title in row["support_titles"]}
    blocked.update(title for _, row in w1.read_lines(manifest)
                   for title in refs[row["dataset"]][row["question_key"]]["titles"])
    rows, counts = w3.build_pool(dump, tests, refs, blocked)
    eligible = select_pool(rows, counts)
    cases = w3.choose_states(tok, test, eligible, counts, blocked, limit=LIMIT)
    if not cases:
        raise ValueError("No independently sampled natural QA cases")
    missing = defaultdict(set)
    for case in cases:
        for title, sentence in case["missing_facts"]:
            missing[title].add(sentence)
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    catalog, digest, size, _, found = w3.scan_corpus(corpus, cases, progress, missing_by_title=missing)
    if (digest, size) != (prior["inputs"]["corpus_sha256"], prior["inputs"]["corpus_rows"]):
        raise ValueError("Use the byte-identical original Wikipedia corpus")
    for case in cases:
        case["queries"], case["visible_titles"] = visible_queries(case["question"], case["visible"], catalog)
        case["strict_corpus_all"] = all(tuple(fact) in found for fact in case["missing_facts"])
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "records": cases, "anchors": old["anchors"],
                       "retriever_id": old["retriever_id"]})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "by_dataset": counts,
               "blocked_support_pages": len(blocked), "prepared_sha256": sha256(path),
               "inputs": {**inputs, "corpus_sha256": digest, "corpus_rows": size,
                          "w3_prepared_sha256": sha256(run_dir / W3_CASES),
                          "w4c_summary_sha256": sha256(run_dir / W4C_SUMMARY)}}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(cases), len(cases), detail="ready")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def score_case(case, returned):
    def score(query):
        return w2.measure(returned[query], case["missing_facts"])["all_missing_facts"]

    row = {key: score(case["queries"][key]) for key in FIXED_ARMS}
    known = score(case["probe_title"])
    visible = any(score(query) for query in case["visible_titles"])
    fixed = any(row.values())
    return {"dataset": case["dataset"], "qid": case["qid"], **row, "strict_corpus_all": case["strict_corpus_all"],
            "known_title": known, "any_fixed": fixed, "any_visible_candidate": visible or fixed,
            "known_title_beyond_fixed": known and not fixed,
            "visible_candidate_beyond_fixed": visible and not fixed,
            "known_title_beyond_visible": known and not (fixed or visible), "returned": returned}


def summarize(rows):
    fields = (*FIXED_ARMS, "strict_corpus_all", "known_title", "any_fixed", "any_visible_candidate",
              "known_title_beyond_fixed", "visible_candidate_beyond_fixed", "known_title_beyond_visible")
    by_dataset = {}
    for dataset in SUPPORT_DATASETS:
        members = [row for row in rows if row["dataset"] == dataset]
        counts = {field: sum(row[field] for row in members) for field in fields}
        by_dataset[dataset] = {"n": len(members), **counts}
    if all(row["n"] >= MIN_ROWS and row["visible_candidate_beyond_fixed"] >= MIN_GAP
           for row in by_dataset.values()):
        decision = "measure_frozen_action_selection"
    elif all(row["n"] >= MIN_ROWS
             and row["known_title"] >= MIN_PROBE
             and row["known_title_beyond_fixed"] >= MIN_GAP for row in by_dataset.values()):
        decision = "one_frozen_query_generation_check"
    else:
        decision = "insufficient_headroom_for_training"
    return by_dataset, decision


def case_queries(case):
    return [*case["queries"].values(), *case["visible_titles"], case["probe_title"]]


def run(args, run_dir, progress):
    path = run_dir / PREPARED
    data, ready = w1.read_json(path), w1.read_json(run_dir / PREP_SUMMARY)
    if (any(obj["protocol"] != PROTOCOL for obj in (data, ready))
            or ready["prepared_sha256"] != sha256(path) or data["retriever_id"] != args.retriever_id):
        raise ValueError("Use the frozen W5a preparation and original retriever")
    cases = data["records"]
    retriever = w2.Retriever(args.retriever)
    progress.update("running", 0, len(cases), detail="checking_retriever")
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w5_natural_retriever_check.json", check)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    rows, queries = [], 0
    for offset in range(0, len(cases), 8):
        batch = cases[offset:offset + 8]
        wanted = list(dict.fromkeys(query for case in batch for query in case_queries(case)))
        returned = dict(zip(wanted, retriever.search(wanted), strict=True))
        queries += len(wanted)
        rows.extend(score_case(case, {query: returned[query] for query in case_queries(case)}) for case in batch)
        progress.update("running", len(rows), len(cases), detail="querying_wikipedia", metrics={"queries": queries})
    by_dataset, decision = summarize(rows)
    atomic_json(run_dir / "w5_natural_cases.json", {"protocol": PROTOCOL, "records": rows})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": check,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "physical_queries_excluding_anchors": queries,
               "by_dataset": by_dataset, "pilot_minimum": {"cases": MIN_ROWS, "probe": MIN_PROBE, "gap": MIN_GAP},
               "decision": decision}
    atomic_json(run_dir / "wiki_rsi_w5_natural_summary.json", summary)
    progress.update("done", len(rows), len(cases), detail="completed", metrics={"queries": queries})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
