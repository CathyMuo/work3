"""Locate W4a losses on the frozen source cohort and test a one-witness repair rule."""

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict

from work3 import wiki_rsi_repair_pool as w4
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w4b_repair_audit_v1"
OLD_SUMMARY = "wiki_rsi_w4_repair_pool_summary.json"
MODES = ("strict_paired", "strict_one_witness", "normalized_paired", "normalized_one_witness")
STAGES = ("source_with_positive", "source_with_prefix_clean_positive",
          "source_with_required_foil", "source_with_clean_pair")
BIRTH_CUE = re.compile(r"\bborn\b|\b\d{4}\s*[-–—−]\s*(?:\S+\s+){0,2}\d{4}\b", re.IGNORECASE)


def clean_text(body):
    normalized = unicodedata.normalize("NFKC", html.unescape(body)).replace("\ufeff", "").replace("\u200b", "")
    return re.sub(r"\s+([,.;])", r"\1", " ".join(normalized.split()))


def lead_names_subject(title, lead):
    title_words, lead_words = text_key(title).split(), text_key(lead).split()
    if len(title_words) < 2 or not 2 <= len(lead_words) <= 8:
        return False
    remaining = iter(lead_words)
    return all(word in remaining for word in title_words)


def birth_entries(title, docs):
    """Read each parser on the same-title passages and retain passage-level witnesses."""
    matches = [doc for doc in docs if title_key(doc[0]) == title_key(title)]
    headers = name_matches = 0
    strict_years, clean_years = {}, {}
    for page, body in matches:
        header = w4.TITLE_AND_DATES.match(body)
        headers += bool(header)
        if header and text_key(header[1]) == text_key(title):
            name_matches += 1
            date = w4.BIRTH.fullmatch(header[2].strip())
            if date:
                year = (date[1] or date[2])[-4:]
                if 1400 <= int(year) <= 2018:
                    strict_years.setdefault(year, [page, body])
        header = w4.TITLE_AND_DATES.match(clean_text(body))
        if header and lead_names_subject(title, header[1]):
            date = w4.BIRTH.fullmatch(header[2].strip())
            if date:
                year = (date[1] or date[2])[-4:]
                if 1400 <= int(year) <= 2018:
                    clean_years.setdefault(year, [page, body])
    clean_years.update(strict_years)
    strict = next(iter(strict_years.items())) if len(strict_years) == 1 else None
    normalized = next(iter(clean_years.items())) if len(clean_years) == 1 else None
    if not docs:
        status = "no_return"
    elif not matches:
        status = "no_exact_title"
    elif strict:
        status = "strict_verified"
    elif len(strict_years) > 1:
        status = "conflicting_strict_birth_years"
    elif not headers:
        status = "no_leading_parenthesis"
    elif not name_matches:
        status = "lead_name_mismatch"
    else:
        status = "no_verified_year_in_parenthesis"
    return {"status": status, "same_title": bool(matches),
            "birth_cue": any(BIRTH_CUE.search(clean_text(body)) for _, body in matches),
            "strict": strict, "normalized": normalized}


def scan_people(corpus, names, progress):
    by_title = defaultdict(list)
    digest = hashlib.sha256()
    before = corpus.stat()
    rows = 0
    progress.update("running", detail="scanning_repair_people")
    with corpus.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            raw, _, body = json.loads(line)["contents"].partition("\n")
            key = title_key(raw)
            if key in names:
                by_title[key].append((raw, body))
            if rows % 100000 == 0:
                progress.update("running", detail="scanning_repair_people", metrics={"corpus_rows": rows})
    after = corpus.stat()
    if not rows or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Wikipedia corpus was empty or changed during audit")
    return by_title, digest.hexdigest(), rows


def candidates(source, prefix, people, returns, evidence, mode, *, work_kind="film"):
    parser = "strict" if mode.startswith("strict") else "normalized"
    paired = mode.endswith("paired")
    stages = set()
    records = []
    for role in sorted(people, key=lambda key: w4.order_key(f"{source}:{key}")):
        target = people[role]
        positive = evidence[target][parser]
        if not positive:
            continue
        stages.add("source_with_positive")
        year, witness = positive
        question = f"What year was the {role} of the {work_kind} {source} born?"
        normalized_question = text_key(question)
        if (w4.has_year(prefix, year) or contains(normalized_question, year)
                or contains(normalized_question, text_key(target))):
            continue
        stages.add("source_with_prefix_clean_positive")
        for other in people.values():
            if title_key(other) == title_key(target):
                continue
            foil_evidence = evidence[other]
            if paired:
                foil_year = foil_evidence[parser]
                if not foil_year or year == foil_year[0]:
                    continue
            elif not foil_evidence["same_title"]:
                continue
            stages.add("source_with_required_foil")
            if (w4.has_year(returns[other], year)
                    or any(title_key(title) == title_key(target) for title, _ in returns[other])):
                continue
            stages.add("source_with_clean_pair")
            records.append({"source": source, "role": role, "question": question, "answer": year,
                            "target": target, "foil": other, "target_witness": witness})
    return stages, records


def choose_candidates(cases):
    """Freeze one exact source/target/foil triple per state across all splits."""
    used = set()
    selected = []
    for options in cases:
        for case in options:
            pages = {title_key(case[key]) for key in ("source", "target", "foil")}
            if pages & used:
                continue
            split = w4.SPLIT_ORDER[len(selected) % len(w4.SPLIT_ORDER)]
            selected.append({**case, "split": split})
            used.update(pages)
            break
        if len(selected) == sum(w4.SPLIT_CAPS.values()):
            break
    return selected


def collect(retriever, sources, blocked, progress):
    histories, people_returns = [], {}
    for offset in range(0, len(sources), 8):
        batch = sources[offset:offset + 8]
        prefixes = retriever.search(batch)
        names = {}
        for source, prefix in zip(batch, prefixes, strict=True):
            people, _ = w4.visible_credits(source, prefix, blocked)
            histories.append({"source": source, "prefix": prefix, "people": people})
            names.update(dict.fromkeys(name for name in people.values() if name not in people_returns))
        queries = list(names)
        people_returns.update(zip(queries, retriever.search(queries), strict=True))
        progress.update("running", len(histories), len(sources), detail="verifying_repair_actions",
                        metrics={"queries": len(histories) + len(people_returns)})
    return histories, people_returns


def analyze_people(live, corpus):
    evidence, counts, by_person = {}, Counter(), {}
    for name, docs in live.items():
        page_docs = corpus.get(title_key(name), [])
        observed, indexed = birth_entries(name, docs), birth_entries(name, page_docs)
        evidence[name] = observed
        counts[f"live_{observed['status']}"] += 1
        counts["corpus_exact_title"] += bool(page_docs)
        counts["corpus_normalized_birth"] += bool(indexed["normalized"])
        counts["corpus_birth_but_title_missed"] += bool(indexed["normalized"] and not observed["same_title"])
        counts["corpus_birth_but_not_in_live_passages"] += bool(
            indexed["normalized"] and observed["same_title"] and not observed["normalized"])
        counts["live_title_with_birth_cue"] += observed["birth_cue"]
        counts["live_strict_birth"] += bool(observed["strict"])
        counts["live_normalized_birth"] += bool(observed["normalized"])
        counts["live_normalized_only_birth"] += bool(observed["normalized"] and not observed["strict"])
        by_person[name] = {"live": observed, "corpus_title_passages": len(page_docs), "corpus": indexed}
    return evidence, counts, by_person


def run(args, run_dir, progress):
    prepared_path = run_dir / w4.PREPARED
    prep_path = run_dir / w4.PREP_SUMMARY
    data, prep = w1.read_json(prepared_path), w1.read_json(prep_path)
    old = w1.read_json(run_dir / OLD_SUMMARY)
    if (any(obj["protocol"] != w4.PROTOCOL for obj in (data, prep, old))
            or prep["prepared_sha256"] != sha256(prepared_path)
            or old["preparation_sha256"] != sha256(prep_path)
            or old["sources_available"] != len(data["sources"])
            or not old["retriever"]["passed"]
            or data["retriever_id"] != args.retriever_id
            or old["retriever"]["retriever_id"] != args.retriever_id):
        raise ValueError("Use the completed W4a run, frozen sources and original retriever")
    if args.corpus is None and args.drzero is None:
        raise ValueError("Set WIKI_RSI_CORPUS or DRZERO_DIR for the original Wikipedia corpus")
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    sources = data["sources"]
    progress.update("running", 0, len(sources), detail="checking_retriever")
    retriever = w2.Retriever(args.retriever)
    passed, group = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": group}
    atomic_json(run_dir / "wiki_rsi_w4_repair_audit_retriever_check.json", check)
    if not passed:
        progress.update("blocked", 0, len(sources), detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    histories, returns = collect(retriever, sources, set(data["blocked"]), progress)
    indexed, digest, count = scan_people(corpus, {title_key(name) for name in returns}, progress)
    if (digest, count) != (prep["corpus_sha256"], prep["corpus_rows"]):
        raise ValueError("Use the byte-identical W4a Wikipedia corpus")
    evidence, person_counts, by_person = analyze_people(returns, indexed)
    by_mode = {key: Counter(dict.fromkeys(STAGES, 0)) for key in MODES}
    options = []
    original_accepted = 0
    for trace in histories:
        source, prefix, people = trace["source"], trace["prefix"], trace["people"]
        original_accepted += w4.repair_case(source, prefix, None, people, returns, set()) is not None
        trace["stages"] = {}
        for mode in MODES:
            stages, records = candidates(source, prefix, people, returns, evidence, mode)
            trace["stages"][mode] = sorted(stages)
            by_mode[mode].update(stages)
            if mode == "normalized_one_witness":
                options.append(records)
    selected = choose_candidates(options)
    by_split = {split: Counter(n=0, question_query_verified_target=0, question_query_year_visible=0)
                for split in w4.SPLIT_CAPS}
    for offset in range(0, len(selected), 8):
        batch = selected[offset:offset + 8]
        responses = retriever.search([case["question"] for case in batch])
        for case, docs in zip(batch, responses, strict=True):
            target_year = birth_entries(case["target"], docs)["normalized"]
            case["original_question"] = {"verified_target": bool(target_year and target_year[0] == case["answer"]),
                                         "year_visible": w4.has_year(docs, case["answer"]), "passages": docs}
            counts = by_split[case["split"]]
            counts["n"] += 1
            counts["question_query_verified_target"] += case["original_question"]["verified_target"]
            counts["question_query_year_visible"] += case["original_question"]["year_visible"]
    source_visible = sum(bool(row["people"]) for row in histories)
    atomic_json(run_dir / "w4_repair_audit_cases.json", {"protocol": PROTOCOL, "source_traces": histories,
                 "person_returns": returns, "person_audit": by_person, "selected": selected})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": check,
               "preparation_sha256": sha256(prep_path), "corpus_sha256": digest, "corpus_rows": count,
               "previous_source_visible": old["counts"].get("source_with_two_visible_credits", 0),
               "previous_person_queries": old["counts"].get("person_queries", 0),
               "source_examined": len(histories), "source_with_two_visible_credits": source_visible,
               "original_rule_accepts_before_split": original_accepted,
               "queries": {"source": len(histories), "person": len(returns), "question": len(selected)},
               "by_person_query": person_counts, "by_mode": by_mode,
               "normalized_one_witness_page_disjoint": len(selected), "by_split": by_split}
    atomic_json(run_dir / "wiki_rsi_w4_repair_audit_summary.json", summary)
    progress.update("done", len(histories), len(sources), detail="completed",
                    metrics={"repair_states": len(selected), "queries": len(histories) + len(returns) + len(selected)})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
