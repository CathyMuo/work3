"""Extend the independent training pool with verified credits from new Wikipedia works."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import zip_longest

from work3 import wiki_rsi_repair_audit as audit
from work3 import wiki_rsi_repair_pool as pool
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import text_key, title_key

TRAIN_CAP = 768
SOURCE_CAP_PER_KIND = 2048
KIND = re.compile(r"\b(?:is|was)\s+(?:an?|the)\s+[^.!?]{0,95}?\b(book|novel|song|single|album)\b", re.IGNORECASE)
CREDIT = re.compile(r"\b(written|illustrated|edited|recorded|performed|composed|produced) by\s+", re.IGNORECASE)
PERSON = re.compile(
    rf"({pool.NAME})(?=\s*[,.;:]|$|\s+(?:and\s+)?(?:written|illustrated|edited|recorded|performed|composed|produced) by\b)"
)
BOOK_ROLES = {"written": "author", "illustrated": "illustrator", "edited": "editor"}
MUSIC_ROLES = {"written": "writer", "recorded": "recording artist", "performed": "performer",
               "composed": "composer", "produced": "producer"}
MARKERS = (b"directed by ", b"written by ", b"produced by ", b"illustrated by ", b"edited by ",
           b"recorded by ", b"performed by ", b"composed by ")


def work_credits(title, body):
    film = pool.film_credits(title, body)
    if len(film) >= 2:
        return "film", film
    lead = body[:480]
    if not text_key(lead).startswith(text_key(title) + " "):
        return None, {}
    kind = KIND.search(lead)
    if not kind:
        return None, {}
    kind = {"novel": "book", "single": "song"}.get(kind[1].lower(), kind[1].lower())
    roles = BOOK_ROLES if kind == "book" else MUSIC_ROLES
    matches = list(CREDIT.finditer(lead))
    counts = Counter(match[1].lower() for match in matches)
    roster = {}
    for match in matches:
        role = match[1].lower()
        tail = lead[match.end():]
        person = PERSON.match(tail)
        if role in roles and counts[role] == 1 and person:
            suffix = tail[person.end():]
            if not re.match(rf",\s*(?:(?:and|(?:along )?with|alongside)\s+)?{pool.NAME}", suffix):
                roster[roles[role]] = person[1]
    return kind, roster


def source_candidates(path, blocked, progress):
    digest, before = hashlib.sha256(), path.stat()
    sources, seen, rows = defaultdict(dict), set(), 0
    progress.update("running", detail="scanning_repair_sources")
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            lower = line.lower()
            if any(marker in lower for marker in MARKERS):
                raw, _, body = json.loads(line)["contents"].partition("\n")
                name, title = raw.strip().strip('"'), title_key(raw)
                if title and title not in blocked and title not in seen:
                    kind, people = work_credits(name, body)
                    if len({title_key(person) for person in people.values()} - blocked - {title}) >= 2:
                        sources[kind][title] = name
                        seen.add(title)
            if rows % 100000 == 0:
                progress.update("running", detail="scanning_repair_sources", metrics={"corpus_rows": rows})
    after = path.stat()
    if not rows or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Wikipedia corpus was empty or changed during the source scan")
    groups = [[(kind, titles[key]) for key in sorted(titles, key=pool.order_key)[:SOURCE_CAP_PER_KIND]]
              for kind, titles in sorted(sources.items())]
    candidates = [pair for group in zip_longest(*groups) for pair in group if pair is not None]
    return candidates, digest.hexdigest(), rows, {kind: len(titles) for kind, titles in sources.items()}


def get_roster(kind, source, docs, blocked):
    excluded = blocked | {title_key(source)}
    for title, body in docs:
        if title_key(title) != title_key(source):
            continue
        observed, people = work_credits(source, body)
        allowed = {role: name for role, name in people.items() if title_key(name) not in excluded}
        if kind == observed and len({title_key(name) for name in allowed.values()}) >= 2:
            return allowed
    return {}


def verify_additions(retriever, candidates, existing, blocked, progress, *, fits_budget):
    selected = []
    used = {title_key(row[key]) for row in existing for key in ("source", "target", "foil")}
    returns, evidence, queries = {}, {}, Counter()
    for offset in range(0, len(candidates), 8):
        batch = candidates[offset:offset + 8]
        prefixes = retriever.search([source for _, source in batch])
        queries["source"] += len(batch)
        roster = [get_roster(kind, source, docs, blocked | used)
                  if not any(title_key(name) in blocked for name, _ in docs) else {}
                  for (kind, source), docs in zip(batch, prefixes, strict=True)]
        names = list(dict.fromkeys(name for people in roster for name in people.values() if name not in returns))
        for name, docs in zip(names, retriever.search(names), strict=True):
            returns[name] = docs
            evidence[name] = audit.birth_entries(name, docs)
        queries["person"] += len(names)
        for (kind, source), prefix, people in zip(batch, prefixes, roster, strict=True):
            if len(existing) + len(selected) >= TRAIN_CAP:
                break
            _, options = audit.candidates(source, prefix, people, returns, evidence,
                                          "normalized_one_witness", work_kind=kind)
            for row in options:
                pages = {title_key(row[key]) for key in ("source", "target", "foil")}
                if len(pages) != 3 or pages & used or pages & blocked:
                    continue
                record = {**row, "work_kind": kind, "visible": w2.normalized_docs(prefix),
                          "positive_query": row["target"], "negative_query": row["foil"],
                          "positive_return": returns[row["target"]], "negative_return": returns[row["foil"]]}
                if not fits_budget(record):
                    continue
                used.update(pages)
                selected.append(record)
                break
        progress.update("running", offset + len(batch), len(candidates), detail="verifying_repair_actions",
                        metrics={"training_pairs": len(existing) + len(selected), "queries": sum(queries.values())})
        if len(existing) + len(selected) >= TRAIN_CAP:
            break
    return selected, queries
