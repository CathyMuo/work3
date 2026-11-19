"""Build page-isolated repair candidates with live evidence from the P0 index."""

import gzip
import hashlib
import json
import re
from collections import Counter

from work3 import wiki_rsi_candidate_audit as w3b
from work3 import wiki_rsi_coverage_audit as w3c
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w4a_repair_pool_v1"
PREPARED = "w4_repair_pool_prepared.json"
PREP_SUMMARY = "wiki_rsi_w4_repair_pool_prepare_summary.json"
W3C_SUMMARY = "wiki_rsi_w3_coverage_audit_summary.json"
SOURCE_CAP = 768
SPLIT_CAPS = {"train": 60, "calibration": 20, "test": 20}
SPLIT_ORDER = ("train", "calibration", "train", "test", "train")
SEED = "20261119"
ROLE_NAMES = {"directed": "director", "written": "writer", "produced": "producer"}
ROLE_MARKER = re.compile(r"\b(directed|written|produced) by\s+", re.IGNORECASE)
NAME = r"[A-Z][a-z][A-Za-z'’\-]*(?:[ \t]+[A-Z][A-Za-z'’\-]*){1,2}"
PERSON = re.compile(
    rf"({NAME})(?=\s*[,.;:]|$|\s+(?:and\s+)?(?:starring|starr(?:ed|s)|(?:directed|produced|written) by)\b)"
)
DATE = r"(?:[A-Za-z]+ \d{1,2},? |\d{1,2} [A-Za-z]+ )?\d{4}"
TITLE_AND_DATES = re.compile(r"^\s*([^()\n;]{2,110})\s*\(([^()\n]{4,110})\)")
BIRTH = re.compile(rf"(?:born\s+({DATE})|({DATE})\s*[–—−-]\s*{DATE})", re.IGNORECASE)


def order_key(value):
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).digest()


def film_credits(title, body):
    lead = body[:480]
    if not text_key(lead).startswith(text_key(title) + " ") or not re.search(r"\bfilm\b", lead, re.IGNORECASE):
        return {}
    markers = list(ROLE_MARKER.finditer(lead))
    counts = Counter(marker[1].lower() for marker in markers)
    credits = {}
    for marker in markers:
        role = marker[1].lower()
        tail = lead[marker.end():]
        name = PERSON.match(tail)
        if not name or counts[role] != 1:
            continue
        suffix = tail[name.end():]
        if re.match(rf",\s*(?:(?:and|(?:along )?with|alongside)\s+)?{NAME}", suffix):
            continue
        credits[ROLE_NAMES[role]] = name[1]
    return credits


def verified_birth(title, docs):
    years = {}
    for page, body in docs:
        if title_key(page) != title_key(title):
            continue
        header = TITLE_AND_DATES.match(body)
        if not header or text_key(header[1]) != text_key(title):
            continue
        date = BIRTH.fullmatch(header[2].strip())
        if date:
            year = (date[1] or date[2])[-4:]
            if 1400 <= int(year) <= 2018:
                years.setdefault(year, [page, body])
    if len(years) == 1:
        return next(iter(years.items()))
    return None


def blocked_titles(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        datasets = json.load(stream)["datasets"]
    titles = {title_key(title) for dataset in datasets.values()
              for item in dataset["items"].values() for title in item["titles"]}
    if not titles:
        raise ValueError("No evaluation support pages to exclude")
    return titles


def scan_sources(path, blocked, progress):
    digest = hashlib.sha256()
    rows = 0
    sources = {}
    snapshot = path.stat()
    progress.update("running", detail="scanning_repair_sources")
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            lower = line.lower()
            if any(marker in lower for marker in (b"directed by ", b"written by ", b"produced by ")):
                name, _, body = json.loads(line)["contents"].partition("\n")
                key = title_key(name)
                if key and key not in blocked and key not in sources:
                    credits = {title_key(person) for person in film_credits(name.strip('"'), body).values()}
                    if len(credits - blocked - {key}) >= 2:
                        sources[key] = name.strip().strip('"')
            if rows % 100000 == 0:
                progress.update("running", detail="scanning_repair_sources", metrics={"corpus_rows": rows})
    after = path.stat()
    if not rows or (snapshot.st_size, snapshot.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("The corpus is empty or changed during source scanning")
    return digest.hexdigest(), rows, sources


def prepare(args, run_dir, progress):
    past_prep = w1.read_json(run_dir / w3c.PREP_SUMMARY)
    past = w1.read_json(run_dir / W3C_SUMMARY)
    anchor_source = w1.read_json(run_dir / w3c.PREPARED)
    if (any(record["protocol"] != w3c.PROTOCOL for record in (past_prep, past, anchor_source))
            or past_prep["prepared_sha256"] != sha256(run_dir / w3c.PREPARED)
            or past["preparation_sha256"] != sha256(run_dir / w3c.PREP_SUMMARY)
            or not past["retriever"]["passed"]
            or past["retriever"]["retriever_id"] != anchor_source["retriever_id"]):
        raise ValueError("Use the completed W3c run and its retriever anchors")
    reference_digest = sha256(args.references)
    w3a = w1.read_json(run_dir / w3b.W3A_PREP_SUMMARY)
    if reference_digest != w3a["inputs"]["references_sha256"]:
        raise ValueError("Use the original evaluation references to isolate support pages")
    blocked = blocked_titles(args.references)
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    digest, rows, sources = scan_sources(corpus, blocked, progress)
    inputs = past_prep["inputs"]
    if (digest, rows) != (inputs["corpus_sha256"], inputs["corpus_rows"]):
        raise ValueError("Use the byte-identical W3c corpus")
    selected = [sources[key] for key in sorted(sources, key=order_key)[:SOURCE_CAP]]
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "sources": selected, "blocked": sorted(blocked),
                       "anchors": anchor_source["anchors"], "retriever_id": anchor_source["retriever_id"]})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "corpus_sha256": digest,
               "corpus_rows": rows, "references_sha256": reference_digest, "blocked_titles": len(blocked),
               "source_eligible": len(sources), "source_selected": len(selected),
               "prepared_sha256": sha256(path)}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(selected), len(selected), detail="ready", metrics={"corpus_rows": rows})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def visible_credits(source, prefix, blocked):
    excluded = blocked | {title_key(source)}
    for page, body in prefix:
        if title_key(page) == title_key(source):
            credits = film_credits(source, body)
            available = {role: person for role, person in credits.items()
                         if title_key(person) not in excluded}
            if len({title_key(person) for person in available.values()}) >= 2:
                return available, [page, body]
    return {}, None


def has_year(docs, year):
    return any(contains(text_key(title + " " + body), year) for title, body in docs)


def repair_case(source, prefix, bridge, people, returned, used):
    if title_key(source) in used:
        return None
    verified = {}
    for role, name in people.items():
        if title_key(name) not in used:
            evidence = verified_birth(name, returned[name])
            if evidence:
                verified[role] = (name, *evidence)
    for role in sorted(verified, key=lambda value: order_key(f"{source}:{value}")):
        person, year, witness = verified[role]
        question = f"What year was the {role} of the film {source} born?"
        normalized = text_key(question)
        if has_year(prefix, year) or contains(normalized, year) or contains(normalized, text_key(person)):
            continue
        for foil_role, (foil, foil_year, foil_witness) in verified.items():
            if (foil_role == role or year == foil_year or title_key(person) == title_key(foil)
                    or has_year(returned[foil], year)
                    or any(title_key(page) == title_key(person) for page, _ in returned[foil])):
                continue
            return {"source": source, "role": role, "question": question, "answer": year,
                    "prefix": prefix, "bridge": bridge,
                    "target": {"query": person, "witness": witness, "passages": returned[person]},
                    "foil": {"query": foil, "year": foil_year, "witness": foil_witness, "passages": returned[foil]}}
    return None


def run(args, run_dir, progress):
    path = run_dir / PREPARED
    data, prep = w1.read_json(path), w1.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or prep["protocol"] != PROTOCOL
            or prep["prepared_sha256"] != sha256(path) or data["retriever_id"] != args.retriever_id):
        raise ValueError("Use the frozen W4a preparation and original retriever")
    sources = data["sources"]
    retriever = w2.Retriever(args.retriever)
    progress.update("running", 0, len(sources), detail="checking_retriever")
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w4_repair_pool_retriever_check.json", check)
    if not passed:
        progress.update("blocked", 0, len(sources), detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    records = []
    blocked = set(data["blocked"])
    used = set()
    counts = Counter()
    split_counts = Counter()
    person_returns = {}
    total = sum(SPLIT_CAPS.values())
    for offset in range(0, len(sources), 8):
        batch = sources[offset:offset + 8]
        prefixes = retriever.search(batch)
        counts["source_queries"] += len(batch)
        roster = [visible_credits(source, prefix, blocked) for source, prefix in zip(batch, prefixes, strict=True)]
        queries = list(dict.fromkeys(name for people, _ in roster for name in people.values()
                                     if name not in person_returns and title_key(name) not in used))
        person_returns.update(zip(queries, retriever.search(queries), strict=True))
        counts["person_queries"] += len(queries)
        accepted = []
        for source, prefix, (people, bridge) in zip(batch, prefixes, roster, strict=True):
            if len(records) + len(accepted) == total:
                break
            counts["source_examined"] += 1
            counts["source_with_two_visible_credits"] += bool(people)
            case = repair_case(source, prefix, bridge, people, person_returns, used)
            if case:
                split = SPLIT_ORDER[(len(records) + len(accepted)) % len(SPLIT_ORDER)]
                if split_counts[split] >= SPLIT_CAPS[split]:
                    raise RuntimeError("Repair pool split caps are inconsistent")
                case["split"] = split
                split_counts[split] += 1
                used.update(title_key(page) for page in (source, case["target"]["query"], case["foil"]["query"]))
                accepted.append(case)
        questions = [case["question"] for case in accepted]
        for case, returned in zip(accepted, retriever.search(questions), strict=True):
            evidence = verified_birth(case["target"]["query"], returned)
            case["original_question"] = {"verified_target": bool(evidence and evidence[0] == case["answer"]),
                                         "year_visible": has_year(returned, case["answer"]), "passages": returned}
        counts["question_queries"] += len(questions)
        records.extend(accepted)
        progress.update("running", counts["source_examined"], len(sources), detail="verifying_repair_actions",
                        metrics={"repair_states": len(records), "queries": sum(counts[key] for key in
                                 ("source_queries", "person_queries", "question_queries"))})
        if len(records) == total:
            break
    by_split = {}
    for split in SPLIT_CAPS:
        members = [case for case in records if case["split"] == split]
        by_split[split] = {"n": len(members), "by_role": Counter(case["role"] for case in members),
                           "question_query_verified_target": sum(case["original_question"]["verified_target"]
                                                                 for case in members),
                           "question_query_year_visible": sum(case["original_question"]["year_visible"]
                                                              for case in members)}
    atomic_json(run_dir / "w4_repair_pool_cases.json", {"protocol": PROTOCOL, "records": records})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": check,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "sources_available": len(sources),
               "counts": counts, "distinct_support_and_foil_pages": len(used), "by_split": by_split}
    atomic_json(run_dir / "wiki_rsi_w4_repair_pool_summary.json", summary)
    progress.update("done", counts["source_examined"], len(sources), detail="completed",
                    metrics={"repair_states": len(records)})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
