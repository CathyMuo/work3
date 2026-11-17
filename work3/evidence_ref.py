"""Build the compact public evaluation annotations for the P0 retrieval diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from work3.p0_diag import normalize

DATASETS = ("hotpotqa", "2wikimultihopqa")
REVISIONS = {"hotpotqa": "e011525", "2wikimultihopqa": "e6c58c6"}
EXPECTED_ROWS = {"hotpotqa": 7405, "2wikimultihopqa": 12576}
NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
SPACE = re.compile(r"\s+")


def text_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    return NON_WORD.sub(" ", text).strip()


def title_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    return SPACE.sub(" ", text.replace("_", " ").strip().strip('"').strip())


def question_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    text = SPACE.sub(" ", text.strip().rstrip("?").strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_keys(answers) -> set[str]:
    return {hashlib.sha256(key.encode("utf-8")).hexdigest() for answer in answers
            if (key := normalize(str(answer)))}


def contains(passage: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {passage} "


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs(value, left, right):
    if isinstance(value, dict):
        if len(value[left]) != len(value[right]):
            raise ValueError(f"Unaligned fields {left} and {right}")
        return zip(value[left], value[right])
    return iter(value)


def prepare_reference(dataset: str, raw: dict) -> dict:
    meta = raw["metadata"]
    context = dict(pairs(meta.get("context") or [], "title",
                         "sentences" if dataset == "hotpotqa" else "content"))
    support: dict[str, set[str]] = {}
    for title, index in pairs(meta["supporting_facts"], "title", "sent_id"):
        sentences = context.get(title, [])
        sentence = text_key(sentences[index]) if 0 <= index < len(sentences) else ""
        support.setdefault(title_key(title), set()).add(sentence)
    if len(support) < 2 or "" in support:
        raise ValueError("Expected at least two annotated support titles")

    qtype = meta["type"]
    aliases = {text_key(a) for a in raw["golden_answers"]}
    aliases -= {"", "yes", "no", "unknown", "none"}
    is_bridge_type = (qtype == "bridge" if dataset == "hotpotqa"
                      else qtype in {"compositional", "inference"})
    answer_title = bridge_title = None
    if is_bridge_type and len(support) == 2 and all("" not in sentences for sentences in support.values()):
        candidates = [title for title, sentences in support.items()
                      if any(contains(s, a) for s in sentences for a in aliases)]
        if len(candidates) == 1:
            answer_title = candidates[0]
            bridge_title = next(title for title in support if title != answer_title)
    return {"answers": sorted(answer_keys(raw["golden_answers"])), "type": qtype,
            "titles": {title: sorted(s) for title, s in sorted(support.items())},
            "answer_title": answer_title, "bridge_title": bridge_title}


def build_dataset(dataset: str, path: Path) -> dict:
    items = {}
    seen = Counter()
    rows = 0
    unannotated = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            raw = json.loads(line)
            key = question_key(raw["question"])
            seen[key] += 1
            # Some source rows can lack usable evidence. Record them; never infer the titles.
            try:
                items[key] = prepare_reference(dataset, raw)
            except (KeyError, ValueError):
                unannotated += 1
    if rows != EXPECTED_ROWS[dataset]:
        raise ValueError(f"{dataset}: expected {EXPECTED_ROWS[dataset]} annotated dev rows, got {rows}")
    unique = {key: row for key, row in items.items() if seen[key] == 1}
    if len(unique) < .90 * rows:
        raise ValueError(f"{dataset}: only {len(unique)} uniquely annotated rows; check source schema")
    return {"revision": REVISIONS[dataset], "source_sha256": sha256(path),
            "rows": rows, "duplicates": sum(n for n in seen.values() if n > 1),
            "unannotated": unannotated, "items": unique}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = {ds: build_dataset(ds, getattr(args, ds)) for ds in DATASETS}
    serial = json.dumps({"schema_version": 1, "datasets": data}, ensure_ascii=False, separators=(",", ":"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gzip.compress(serial.encode("utf-8"), mtime=0))
    for ds, meta in data.items():
        roles = sum(row["bridge_title"] is not None for row in meta["items"].values())
        print(f"{ds}: {len(meta['items'])} unique annotated questions, {roles} role-eligible")


if __name__ == "__main__":
    main()
