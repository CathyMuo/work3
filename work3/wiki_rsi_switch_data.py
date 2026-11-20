"""Mine verified two-fact comparisons from the fixed Wikipedia retriever."""

import hashlib
import json
import re
from collections import Counter

from work3 import wiki_rsi_natural as natural
from work3 import wiki_rsi_repair_pool as pool
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_two_query as w7a
from work3 import wiki_rsi_two_query_live as w7b
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_evidence_switch_v1"
PREPARED = "switch_prepared.json"
PREP_SUMMARY = "wiki_rsi_switch_prepare_summary.json"
SOURCES_PER_RELATION = 4096
CAPS = {"train": 240, "test": 48}
MIN_TRAIN, MIN_TEST = 64, 48
YEAR = r"((?:1[5-9]|20)\d{2})"
DATE = rf"(?:[A-Za-z]+ \d{{1,2}},? |\d{{1,2}} [A-Za-z]+ )?{YEAR}"
SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
MARKER = re.compile(rb"\b(?:founded|established|opened|published|released) (?:in|on|to the public in) ")
RELATIONS = {
    "founding": (re.compile(rf"\b(?:was|were) (?:founded|established) (?:in|on) {DATE}\b", re.I), "founded"),
    "opening": (re.compile(rf"\b(?:was (?:officially )?)?opened (?:to the public in|in|on) {DATE}\b", re.I), "opened"),
    "publication": (re.compile(rf"\b(?:was|were) (?:first )?(?:published|released) (?:in|on) {DATE}\b", re.I),
                    "first published or released"),
}


def sentences(body):
    return SENTENCES.split(body.strip()[:1200])[:5]


def short_title(title):
    return re.sub(r"\s+\([^()]*\)$", "", title)


def mentions_subject(sentence, title, lead):
    prefix = text_key(lead)
    needle = text_key(short_title(title))
    names_page = prefix.startswith(needle + " ") or prefix.startswith("the " + needle + " ")
    return names_page and (contains(text_key(sentence), needle) or re.match(r"^It\s", sentence))


def source_years(title, body, pattern):
    pieces = sentences(body)
    values = {}
    for sentence in pieces:
        if mentions_subject(sentence, title, pieces[0]):
            for match in pattern.finditer(sentence):
                year = int(match[1])
                if 1500 <= year <= 2018:
                    values[year] = sentence if len(sentence.split()) <= 48 else match[0]
    return values


def scan(path, excluded, progress):
    digest, count = hashlib.sha256(), 0
    selected = {relation: {} for relation in RELATIONS}
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            count += 1
            if MARKER.search(line.lower()):
                raw, _, body = json.loads(line)["contents"].partition("\n")
                title = raw.strip().strip('"')
                key = title_key(title)
                if key and key not in excluded and 3 <= len(title) <= 90 and not re.search(r"\b\d{4}\b", title):
                    for relation, (pattern, _) in RELATIONS.items():
                        if key not in selected[relation]:
                            values = source_years(title, body, pattern)
                            if len(values) == 1:
                                selected[relation][key] = (title, next(iter(values)))
            if count % 100000 == 0:
                progress.update("running", detail="mining_two_fact_questions", metrics={"corpus_rows": count})
    return ({rel: [items[key] for key in sorted(items, key=pool.order_key)[:SOURCES_PER_RELATION]]
             for rel, items in selected.items()}, digest.hexdigest(), count)


def observed_fact(title, year, relation, docs):
    pattern, _ = RELATIONS[relation]
    witnesses, neutral = {}, []
    for page, body in docs:
        if title_key(page) != title_key(title):
            continue
        witnesses.update(source_years(title, body, pattern))
        neutral.extend(sentence for sentence in sentences(body)
                       if 4 <= len(sentence.split()) <= 48 and not re.search(r"\b\d{4}\b", sentence))
    if set(witnesses) != {year} or not neutral:
        return None
    witness = witnesses[year]
    baseline = min(neutral, key=lambda text: abs(len(text.split()) - len(witness.split())))
    # Pad the witness with the same original sentence in both states. Both titles remain visible.
    return {"title": title, "year": year, "witness": witness, "neutral": baseline, "returned": docs}


def verified_pair(left, right, relation):
    from work3 import wiki_rsi_evidence_switch as switch

    entries = [left, right]
    first, second = [source["title"] for source in entries]
    if text_key(short_title(first)) == text_key(short_title(second)):
        return None
    _, verb = RELATIONS[relation]
    key = pool.order_key(first + "\0" + second)
    names = [first, second] if key[0] % 2 else [second, first]
    comparison = "earlier" if key[1] % 2 else "later"
    question = f"Which was {verb} {comparison}, {names[0]} or {names[1]}?"
    best = min if comparison == "earlier" else max
    answer = best(entries, key=lambda source: source["year"])["title"]
    branches = []
    for known, entry in enumerate(entries):
        docs = [(source["title"], source["neutral"] + (" " + source["witness"] if i == known else ""))
                for i, source in enumerate(entries)]
        if key[2] % 2:
            docs.reverse()
        branches.append({"query": entry["query"], "witness": (entry["title"], entry["witness"]),
                         "observation": docs, "returned": entry["returned"]})
    example = switch.verified_example(question, [], branches)
    if example is None:
        return None
    return {**example, "aliases": list(dict.fromkeys([answer, short_title(answer)])),
            "relation": relation, "pages": [first, second],
            "witnesses": [branch["witness"] for branch in branches],
            "returns": [entry["returned"] for entry in entries]}


def near_years(subset):
    by_year = {}
    for row in subset:
        by_year.setdefault(row["year"], []).append(row)
    years = sorted(by_year)
    for index, year in enumerate(years):
        for next_year in years[index + 1:]:
            if next_year - year > 35 or not by_year[year]:
                break
            while by_year[year] and by_year[next_year]:
                yield by_year[year].pop(), by_year[next_year].pop()


def build_pairs(retriever, source_groups, progress):
    sources, records, used = Counter(), {"train": [], "test": []}, set()
    queries = 0
    for relation, group in source_groups.items():
        verified = []
        _, verb = RELATIONS[relation]
        for offset in range(0, len(group), 16):
            batch = [(title, year) for title, year in group[offset:offset + 16] if title_key(title) not in used]
            probes = [f"{title} {verb} date" for title, _ in batch]
            for (title, year), query, docs in zip(batch, probes, retriever.search(probes), strict=True):
                row = observed_fact(title, year, relation, docs)
                if row is not None:
                    verified.append({**row, "query": query})
            queries += len(probes)
            progress.update("running", detail="verifying_two_fact_queries", metrics={"queries": queries})
        sources[relation] = len(verified)
        for split, cap in CAPS.items():
            subset = sorted((row for row in verified if
                             (pool.order_key(title_key(row["title"]))[0] % 5 == 0) == (split == "test")),
                            key=lambda row: (row["year"], pool.order_key(row["title"])))
            count = 0
            for left, right in near_years(subset):
                if count >= cap:
                    break
                if used.intersection(title_key(row["title"]) for row in (left, right)):
                    continue
                row = verified_pair(left, right, relation)
                if row is not None:
                    used.update(title_key(title) for title in row["pages"])
                    records[split].append(row)
                    count += 1
    return records, sources, queries


def prepare(args, run_dir, progress):
    analysis, ready = (replay.read_json(run_dir / name) for name in (w7b.SUMMARY, w7b.PREP_SUMMARY))
    natural_prep = replay.read_json(run_dir / natural.PREP_SUMMARY)
    if (analysis["protocol"] != w7b.PROTOCOL or ready["protocol"] != w7b.PROTOCOL
            or analysis["preparation_sha256"] != sha256(run_dir / w7b.PREP_SUMMARY)
            or ready["w7a_sha256"] != sha256(run_dir / w7a.SUMMARY)
            or ready["prepared_sha256"] != sha256(run_dir / w7b.PREPARED)
            or natural_prep["inputs"]["references_sha256"] != sha256(args.references)
            or not args.retriever or args.retriever_id != analysis["retriever_id"]):
        raise ValueError("Use the pinned W7b inputs and original Wikipedia index")
    data = replay.read_json(run_dir / w7b.PREPARED)
    replay.verify_checkpoint(args, ready)
    retriever = wiki.Retriever(args.retriever)
    passed, anchors = wiki.check_anchors(retriever, data["anchors"])
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        print(json.dumps({"passed": passed, "anchors": anchors}, indent=2), flush=True)
        return 2
    excluded = pool.blocked_titles(args.references)
    excluded.update(title_key(title) for row in data["cases"] for title, _ in row["visible"])
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    groups, digest, count = scan(corpus, excluded, progress)
    if (digest, count) != (ready["corpus_sha256"], ready["corpus_rows"]):
        raise ValueError("Use the exact Wikipedia corpus checked in W7b")
    records, verified, queries = build_pairs(retriever, groups, progress)
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "records": records,
                       "retriever_id": args.retriever_id, "anchors": data["anchors"]})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT), "prepared_sha256": sha256(path),
               "w7b_sha256": sha256(run_dir / w7b.SUMMARY), "model_sha256": ready["model_sha256"],
               "corpus_sha256": digest, "corpus_rows": count, "physical_queries_excluding_anchors": queries,
               "verified_entities": verified, "pairs": {split: Counter(row["relation"] for row in rows)
                                                       for split, rows in records.items()},
               "ready_for_reader": len(records["train"]) >= MIN_TRAIN and len(records["test"]) >= MIN_TEST
                                   and len({row["relation"] for row in records["train"]}) >= 2}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", sum(map(len, records.values())), sum(map(len, records.values())), detail="ready")
    print(json.dumps(summary, indent=2), flush=True)
    return 0
