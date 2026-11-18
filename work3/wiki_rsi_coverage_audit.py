"""Audit what strict Wikipedia evidence coverage leaves out on the frozen W3b states."""

import hashlib
import json
from collections import Counter, defaultdict

from work3 import wiki_rsi_candidate_audit as w3b
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w3c_coverage_audit_v1"
PREPARED = "w3_coverage_audit_prepared.json"
PREP_SUMMARY = "wiki_rsi_w3_coverage_audit_prepare_summary.json"
PREVIOUS_SUMMARY = "wiki_rsi_w3_candidate_audit_summary.json"
PREVIOUS_CASES = "w3_candidate_audit_cases.json"


def scan_corpus(path, expected, progress):
    """Find exact strings, with each alternative match kept separate from strict coverage."""
    by_title = defaultdict(set)
    absent = defaultdict(set)
    for (title, sentence), exact in expected.items():
        by_title[title].add(sentence)
        if not exact:
            absent[sentence].add(title)
    words = {sentence: set(sentence.split()) for sentence in absent}
    frequency = Counter(word for tokens in words.values() for word in tokens)
    anchors = defaultdict(list)
    for sentence, tokens in words.items():
        anchor = min(tokens, key=lambda token: (frequency[token], -len(token), token))
        anchors[anchor].append(sentence)
    strict = set()
    boundary = {}
    other_title = {}
    previous_title, previous_body = "", ""
    digest = hashlib.sha256()
    snapshot = path.stat()
    rows = 0
    progress.update("running", detail="scanning_coverage_alternatives")
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            raw_title, _, raw_body = json.loads(line)["contents"].partition("\n")
            title = title_key(raw_title)
            required = by_title.get(title)
            body = text_key(raw_body) if required or anchors else ""
            if required:
                strict.update((title, sentence) for sentence in required if contains(body, sentence))
                if title == previous_title:
                    joined = previous_body + " " + body
                    for sentence in required:
                        fact = title, sentence
                        if (not expected[fact] and fact not in boundary and contains(joined, sentence)
                                and not contains(previous_body, sentence) and not contains(body, sentence)):
                            boundary[fact] = [rows - 1, rows]
            if anchors and title:
                for anchor in set(body.split()).intersection(anchors):
                    found_sentences = []
                    for sentence in anchors[anchor]:
                        remaining = absent[sentence]
                        if remaining == {title} or not contains(body, sentence):
                            continue
                        for wanted_title in remaining - {title}:
                            other_title[wanted_title, sentence] = {"title": title, "row": rows}
                        remaining.intersection_update({title})
                        if not remaining:
                            found_sentences.append(sentence)
                    if found_sentences:
                        anchors[anchor] = [sentence for sentence in anchors[anchor]
                                           if sentence not in found_sentences]
                        if not anchors[anchor]:
                            del anchors[anchor]
            previous_title, previous_body = (title, body) if required else ("", "")
            if rows % 100000 == 0:
                progress.update("running", detail="scanning_coverage_alternatives", metrics={"corpus_rows": rows})
    end = path.stat()
    if not rows or (snapshot.st_size, snapshot.st_mtime_ns) != (end.st_size, end.st_mtime_ns):
        raise ValueError("The corpus is empty or changed during scanning")
    if strict != {fact for fact, exact in expected.items() if exact}:
        raise ValueError("W3b same-title exact coverage did not reproduce")
    return digest.hexdigest(), rows, boundary, other_title


def group_counts(records, fields):
    result = {}
    for dataset in SUPPORT_DATASETS:
        rows = [row for row in records if row["dataset"] == dataset]
        groups = {"all_states": rows, "one_missing_page": [row for row in rows if row["missing_pages"] == 1],
                  "multiple_missing_pages": [row for row in rows if row["missing_pages"] > 1]}
        result[dataset] = {
            group: {"n": len(members), **{field: sum(row[field] for row in members) for field in fields}}
            for group, members in groups.items()
        }
    return result


def case_coverage(case, boundary, other_title):
    facts = []
    for ref in case["corpus"]:
        fact = ref["title"], ref["sentence"]
        facts.append({**ref, "adjacent_rows": boundary.get(fact), "other_title_example": other_title.get(fact)})
    strict = [fact["exact_match"] for fact in facts]
    boundary_hits = [bool(fact["adjacent_rows"]) for fact in facts]
    other_hits = [bool(fact["other_title_example"]) for fact in facts]
    return {
        "dataset": case["dataset"], "qid": case["qid"], "probe_title": case["probe_title"], "facts": facts,
        "missing_pages": len({fact["title"] for fact in facts}), "missing_facts": len(facts),
        "strict_facts": sum(strict), "boundary_facts": sum(boundary_hits), "other_title_facts": sum(other_hits),
        "strict_any": any(strict), "strict_all": all(strict),
        "any_alternative": any(boundary_hits) or any(other_hits),
        "strict_or_boundary_all": all(exact or adjacent for exact, adjacent in zip(strict, boundary_hits, strict=True)),
        "strict_or_other_title_all": all(exact or other for exact, other in zip(strict, other_hits, strict=True)),
        "strict_or_alternative_all": all(exact or adjacent or other
                                        for exact, adjacent, other in zip(strict, boundary_hits, other_hits, strict=True)),
    }


def prepare(args, run_dir, progress):
    started = w1.read_json(run_dir / w3b.PREP_SUMMARY)
    finished = w1.read_json(run_dir / PREVIOUS_SUMMARY)
    w3b_path = run_dir / w3b.PREPARED
    frozen = w1.read_json(w3b_path)
    if (any(record["protocol"] != w3b.PROTOCOL for record in (started, finished, frozen))
            or started["prepared_sha256"] != sha256(w3b_path)
            or finished["preparation_sha256"] != sha256(run_dir / w3b.PREP_SUMMARY)
            or not finished["retriever"]["passed"]
            or finished["retriever"]["retriever_id"] != frozen["retriever_id"]):
        raise ValueError("Use the completed W3b run and its frozen preparation")
    cases = frozen["records"]
    counts = Counter(case["dataset"] for case in cases)
    pairs = {(case["dataset"], case["qid"]) for case in cases}
    if (len(cases) != len(pairs) or counts != started["by_dataset"]
            or any(counts[ds] != finished["by_dataset"][ds]["all_states"]["n"] for ds in SUPPORT_DATASETS)):
        raise ValueError("W3b case set changed")
    expected = {}
    for case in cases:
        for ref in case["corpus"]:
            fact, exact = (ref["title"], ref["sentence"]), ref["exact_match"]
            if not all(fact) or expected.setdefault(fact, exact) != exact:
                raise ValueError("Inconsistent reference in the W3b corpus scan")
    if not expected:
        raise ValueError("W3b has no evidence to audit")
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    digest, count, boundary, other_title = scan_corpus(corpus, expected, progress)
    if digest != started["inputs"]["corpus_sha256"] or count != started["inputs"]["corpus_rows"]:
        raise ValueError("Use the byte-identical W3b corpus")
    records = [case_coverage(case, boundary, other_title) for case in cases]
    fields = ("missing_facts", "strict_facts", "boundary_facts", "other_title_facts", "strict_any", "strict_all",
              "any_alternative", "strict_or_boundary_all", "strict_or_other_title_all", "strict_or_alternative_all")
    previous_cases_sha256 = sha256(run_dir / PREVIOUS_CASES)
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "records": records, "anchors": frozen["anchors"],
                       "retriever_id": frozen["retriever_id"], "w3b_cases_sha256": previous_cases_sha256})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "prepared_sha256": sha256(path),
               "inputs": {"corpus_sha256": digest, "corpus_rows": count, "w3b_prepared_sha256": sha256(w3b_path),
                          "w3b_summary_sha256": sha256(run_dir / PREVIOUS_SUMMARY),
                          "w3b_cases_sha256": previous_cases_sha256},
               "by_dataset": group_counts(records, fields)}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(records), len(records), detail="ready", metrics=counts)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def missing_titles(case):
    return sorted({fact["title"] for fact in case["facts"]})


def score_probes(case, returned, previous):
    missing = [(fact["title"], fact["sentence"]) for fact in case["facts"]]
    titles = missing_titles(case)
    matched = {title: w2.measure(returned[title], missing) for title in titles}
    target = case["probe_title"]
    single = max(outcome["matched_facts"] for outcome in matched.values())
    union_docs = list(dict.fromkeys(doc for title in titles for doc in returned[title]))
    union = w2.measure(union_docs, missing)
    return {"dataset": case["dataset"], "qid": case["qid"], "missing_pages": case["missing_pages"],
            "strict_corpus_all": case["strict_all"], "logical_queries": len(titles),
            "w3b_primary_exact_ordered": w2.normalized_docs(returned[target]) == w2.normalized_docs(previous),
            "primary_any": matched[target]["any_missing_fact"], "primary_all": matched[target]["all_missing_facts"],
            "best_single_any": single > 0, "best_single_all": single == len(missing),
            "union_any": union["any_missing_fact"], "union_all": union["all_missing_facts"],
            "union_adds_facts": union["matched_facts"] > single,
            "union_completes_beyond_single": union["all_missing_facts"] and single < len(missing),
            "union_all_outside_strict_corpus": union["all_missing_facts"] and not case["strict_all"],
            "per_title": matched, "returned": {title: returned[title] for title in titles}}


def run(args, run_dir, progress):
    path = run_dir / PREPARED
    data = w1.read_json(path)
    ready = w1.read_json(run_dir / PREP_SUMMARY)
    previous_path = run_dir / PREVIOUS_CASES
    if (data["protocol"] != PROTOCOL or ready["protocol"] != PROTOCOL
            or ready["prepared_sha256"] != sha256(path) or data["retriever_id"] != args.retriever_id
            or data["w3b_cases_sha256"] != sha256(previous_path)):
        raise ValueError("Use the frozen W3c preparation and original retriever")
    previous = w1.read_json(previous_path)
    lookup = {(row["dataset"], row["qid"]): row["queries"] for row in previous["records"]}
    cases = data["records"]
    if (previous["protocol"] != w3b.PROTOCOL or len(lookup) != len(previous["records"])
            or lookup.keys() != {(row["dataset"], row["qid"]) for row in cases}):
        raise ValueError("W3b case outputs differ from W3c preparation")
    progress.update("running", 0, len(cases), detail="checking_retriever")
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w3_coverage_audit_retriever_check.json", check)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    records, physical_queries = [], 0
    for offset in range(0, len(cases), 8):
        batch = cases[offset:offset + 8]
        queries = list(dict.fromkeys(title for case in batch for title in missing_titles(case)))
        returned = dict(zip(queries, retriever.search(queries), strict=True))
        physical_queries += len(queries)
        for case in batch:
            old = lookup[case["dataset"], case["qid"]][case["probe_title"]]["passages"]
            records.append(score_probes(case, returned, old))
        progress.update("running", len(records), len(cases), detail="querying_wikipedia",
                        metrics={"queries": physical_queries})
    fields = ("strict_corpus_all", "logical_queries", "w3b_primary_exact_ordered", "primary_any", "primary_all",
              "best_single_any", "best_single_all", "union_any", "union_all", "union_adds_facts",
              "union_completes_beyond_single", "union_all_outside_strict_corpus")
    atomic_json(run_dir / "w3_coverage_audit_cases.json", {"protocol": PROTOCOL, "records": records})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": check,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "physical_queries_excluding_anchors": physical_queries,
               "by_dataset": group_counts(records, fields)}
    atomic_json(run_dir / "wiki_rsi_w3_coverage_audit_summary.json", summary)
    progress.update("done", len(records), len(cases), detail="completed", metrics={"queries": physical_queries})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
