"""Repair the Wikipedia RSI reference pack from the two official HotpotQA dev contexts."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from work3.evidence_ref import EXPECTED_ROWS, question_key, sha256, text_key, title_key

DATA = Path(__file__).resolve().parents[1] / "research/data"
V1 = DATA / "evidence_dev_v1.json.gz"
WIKI_RSI_REFERENCES = DATA / "evidence_wiki_rsi_v2.json.gz"
SOURCES = {
    "v1": "4329d003f2ea9cfe590c597c131f9fdc85b36b0e6d52b1614595b02796c9e338",
    "fullwiki": "2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618",
    "distractor": "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
}


def indexed(rows):
    result = {row["_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate HotpotQA official ID")
    return result


def official_sentences(row):
    context = dict(row["context"])
    facts = {}
    for title, index in row["supporting_facts"]:
        sentences = context.get(title, [])
        text = text_key(sentences[index]) if 0 <= index < len(sentences) else ""
        facts.setdefault(title_key(title), set()).add(text)
    return {title: sorted(sentences) for title, sentences in sorted(facts.items())}


def has_gap(titles):
    return any("" in sentences for sentences in titles.values())


def repair_hotpot(items, fullwiki, distractor):
    complete_context = indexed(distractor)
    fullwiki = indexed(fullwiki)
    if fullwiki.keys() != complete_context.keys():
        raise ValueError("Official HotpotQA dev variants have different IDs")
    rebuilt = {}
    counts = Counter()
    for oid, row in fullwiki.items():
        partner = complete_context[oid]
        if any(row[field] != partner[field]
               for field in ("question", "answer", "supporting_facts", "type")):
            raise ValueError(f"Official HotpotQA fields do not agree for {oid}")
        key = question_key(row["question"])
        if key not in items or key in rebuilt:
            raise ValueError(f"Question does not align uniquely with the frozen pack for {oid}")
        original = items[key]
        titles = original["titles"]
        fixed = official_sentences(partner)
        if (titles != official_sentences(row) or original["type"] != row["type"]
                or titles.keys() != fixed.keys()
                or any(not set(sentences) - {""} <= set(fixed[title])
                       for title, sentences in titles.items())):
            raise ValueError(f"Official support facts diverge from the frozen pack for {oid}")
        if not has_gap(titles) and titles != fixed:
            raise ValueError(f"Complete support changed for {oid}")
        counts["old_incomplete"] += has_gap(titles)
        counts["new_incomplete"] += has_gap(fixed)
        counts["rows_changed"] += titles != fixed
        rebuilt[key] = {**original, "titles": fixed}
    if rebuilt.keys() != items.keys():
        raise ValueError("Official HotpotQA dev does not cover the frozen question set")
    return rebuilt, dict(counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=V1)
    parser.add_argument("--fullwiki", type=Path, required=True)
    parser.add_argument("--distractor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Use a new output path")
    for name, expected in SOURCES.items():
        if sha256(getattr(args, name)) != expected:
            raise ValueError(f"{name} source SHA-256 differs from the pinned input")
    with gzip.open(args.v1, "rt", encoding="utf-8") as stream:
        pack = json.load(stream)
    if pack["schema_version"] != 1 or len(pack["datasets"]["hotpotqa"]["items"]) != EXPECTED_ROWS["hotpotqa"]:
        raise ValueError("Unexpected frozen HotpotQA schema or row count")
    fullwiki, distractor = (json.loads(path.read_text(encoding="utf-8"))
                           for path in (args.fullwiki, args.distractor))
    hotpot = pack["datasets"]["hotpotqa"]
    items, counts = repair_hotpot(hotpot["items"], fullwiki, distractor)
    hotpot["items"] = items
    hotpot["repair"] = {"official_dev_context": "distractor_v1",
                        "sources_sha256": SOURCES, **counts}
    serialized = json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    args.out.write_bytes(gzip.compress(serialized.encode("utf-8"), mtime=0))
    print(json.dumps({**counts, "output_sha256": sha256(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
