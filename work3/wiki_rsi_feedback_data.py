"""Freeze page-disjoint evaluation and build paired evidence-grounded supervision."""

import gzip
import hashlib
import json
import re
import time
from collections import Counter
from random import Random

from work3 import wiki_rsi_action_pilot as history
from work3 import wiki_rsi_feedback_protocol as protocol
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_two_query_live as old
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import answer_keys, contains, question_key, sha256, text_key, title_key
from work3.p0_retrieval_diag import question_of, revision, test_questions
from work3.wiki_rsi_audit import support_facts
from work3.wiki_rsi_progress import atomic_json

LOCKED = "feedback_locked.json"
PAIRS = "feedback_pairs.json"


def time_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Frozen phase budget exhausted")


def batches(rows, size=protocol.BATCH):
    for offset in range(0, len(rows), size):
        yield rows[offset:offset + size]


def choose_splits(pool, blocked):
    used, seen = {}, set()
    blocked = set(blocked)
    output = {}
    for split, n in protocol.SPLITS.items():
        selected = []
        for ds in protocol.DATASETS:
            owner = split, ds
            order = sorted(pool[ds], key=lambda row: hashlib.sha256(
                f"{protocol.SEEDS[0]}:{split}:{row['qid']}".encode()).digest())
            current = []
            for row in order:
                pages = set(row["pages"])
                if row["qid"] in seen or blocked & pages or any(used.get(page, owner) != owner for page in pages):
                    continue
                used.update((page, owner) for page in pages)
                seen.add(row["qid"])
                current.append(row)
                if len(current) == n:
                    break
            selected.extend(cluster_cases(current))
        output[split] = selected
    return output


def cluster_cases(rows):
    parents = {page: page for row in rows for page in row["pages"]}

    def find(page):
        while parents[page] != page:
            parents[page] = parents[parents[page]]
            page = parents[page]
        return page

    for row in rows:
        roots = sorted({find(page) for page in row["pages"]})
        for other in roots[1:]:
            parents[other] = roots[0]
    return [{**row, "cluster": find(row["pages"][0])} for row in rows]


def frozen_evaluation(test, refs, blocked):
    answers, _ = test_questions(test)
    prompts = replay.initial_messages(test, {(ds, key) for ds, items in refs.items() for key in items})
    pool = {ds: [] for ds in protocol.DATASETS}
    for (ds, key), messages in prompts.items():
        item = refs[ds][key]
        pages, aliases = sorted(item["titles"]), answers[ds].get(key, ())
        question = question_of(messages[-1]["content"])
        facts = sorted(support_facts(item))
        if (len(pages) >= 2 and not any(not span for _, span in facts) and
                set(item["answers"]) & answer_keys(aliases) and question_key(question) == key):
            pool[ds].append({"dataset": ds, "qid": key, "question": question, "aliases": aliases,
                             "pages": pages, "facts": facts})
    return choose_splits(pool, blocked)


def sample_titles(corpus, blocked, deadline, progress):
    rng, digest = Random(protocol.SEEDS[0]), hashlib.sha256()
    candidates = {}
    signature = corpus.stat()
    with corpus.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            digest.update(line)
            if rng.random() < 0.002 and line.strip():
                title = json.loads(line)["contents"].split("\n", 1)[0].strip()
                key = title_key(title)
                if key and key not in blocked and 4 <= len(title) <= 100:
                    candidates.setdefault(key, title)
            if number % 100000 == 0:
                time_check(deadline)
                progress.update("running", detail="feedback_scan", metrics={"corpus_rows": number})
    after = corpus.stat()
    if (signature.st_size, signature.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Corpus changed during the source scan")
    ordered = sorted(candidates.values())
    sizes = min(protocol.PROPOSALS, len(ordered))
    return rng.sample(ordered, sizes), digest.hexdigest()


def lock(args, folder, progress, deadline):
    source = args.original_run.resolve()
    prepared = replay.read_json(source / old.PREPARED)
    report = replay.read_json(source / old.PREP_SUMMARY)
    if (prepared["protocol"] != old.PROTOCOL or report["protocol"] != old.PROTOCOL or
            report["prepared_sha256"] != sha256(source / old.PREPARED) or
            prepared["retriever_id"] != args.retriever_id):
        raise ValueError("Use the pinned source run, references, and index")
    _, test = replay.input_files(args)
    natural = replay.read_json(source / "wiki_rsi_w5_natural_prepare_summary.json")
    inputs = {"test_sha256": sha256(test), "references_sha256": sha256(args.references)}
    if any(natural["inputs"][key] != value for key, value in inputs.items()):
        raise ValueError("Test parquet or reference bundle differs from the source run")
    retriever = wiki.Retriever(args.retriever)
    parity, anchors = wiki.check_anchors(retriever, prepared["anchors"])
    if not parity:
        return {"ready": False, "reason": "retriever_parity", "anchor_checks": anchors}
    progress.update("running", detail="feedback_models")
    models = {label: replay.model_fingerprint(path) for label, path in
              (("early", args.early_model), ("late", args.late_model))}
    time_check(deadline)
    if models["late"] != report["model_sha256"] or models["early"] == models["late"]:
        raise ValueError("Use the original R3 checkpoint and a distinct early checkpoint")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                       "chat_template.jinja", "special_tokens_map.json", "added_tokens.json")
    if any(models["early"].get(name) != models["late"].get(name) for name in tokenizer_files):
        raise ValueError("Both checkpoints must use the same frozen tokenizer")
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = {ds: block["items"] for ds, block in json.load(stream)["datasets"].items()}
    eval_blocked, train_blocked = history.blocked_history(source, refs)
    eval_blocked.update(title_key(page) for row in prepared["cases"] for page in row["support_titles"])
    train_blocked.update(eval_blocked)
    progress.update("running", detail="feedback_split")
    splits = frozen_evaluation(test, refs, eval_blocked)
    counts = {split: dict(Counter(r["dataset"] for r in rows)) for split, rows in splits.items()}
    clusters = {split: {ds: len({row["cluster"] for row in rows if row["dataset"] == ds})
                        for ds in protocol.DATASETS} for split, rows in splits.items()}
    if any(counts[split].get(ds) != n or clusters[split][ds] < protocol.MIN_CLUSTERS[split]
           for split, n in protocol.SPLITS.items() for ds in protocol.DATASETS):
        return {"ready": False, "reason": "insufficient_holdout", "by_split": counts, "clusters": clusters}
    for split, rows in splits.items():
        done = 0
        for batch in batches(rows, 32):
            time_check(deadline)
            for row, docs in zip(batch, retriever.search([r["question"] for r in batch]), strict=True):
                row["first"] = docs
            done += len(batch)
            progress.update("running", done, len(rows), detail=f"feedback_cache_{split}")
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    titles, digest = sample_titles(corpus, train_blocked, deadline, progress)
    inputs["corpus_sha256"] = digest
    if digest != report["corpus_sha256"]:
        raise ValueError("Use the same fixed Wikipedia corpus as the source run")
    if len(titles) < protocol.PROPOSALS:
        return {"ready": False, "reason": "insufficient_sources", "sources": len(titles)}
    atomic_json(folder / LOCKED, {"protocol": protocol.PROTOCOL, "source": inputs, "retriever_id": args.retriever_id,
                                  "anchors": prepared["anchors"], "splits": splits,
                                  "titles": titles, "blocked_train_pages": sorted(train_blocked),
                                  "models": models, "drzero_revision": revision(args.drzero)})
    return {"ready": True, "locked_sha256": sha256(folder / LOCKED), "by_split": counts, "clusters": clusters,
            "sources": len(titles), "anchor_checks": anchors, "inputs": inputs,
            "shared_first_queries": sum(map(len, splits.values())),
            "models": models, "drzero_revision": revision(args.drzero)}


def answer_field(raw):
    try:
        text = raw.strip()
        if text.startswith("```") and text.endswith("```") and "\n" in text:
            text = text.split("\n", 1)[1][:-3].strip()
        value = json.loads(text)
    except (ValueError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def clip(docs):
    return [(title, body[:900]) for title, body in docs]


def has(docs, title, fact):
    return wiki.measure(docs, [(title_key(title), text_key(fact))])["all_missing_facts"]


def checked_pair(proposal, source, bridge, first, second, family):
    fields = [proposal.get(key) for key in ("question", "answer", "first_quote", "second_quote")]
    if not all(isinstance(value, str) and value.strip() for value in fields):
        return None
    question, answer, first_quote, second_quote = map(str.strip, fields)
    if (not 24 <= len(question) <= 220 or not question.endswith("?") or
            not 2 <= len(answer) <= 72 or text_key(answer) in {"yes", "no", "unknown"} or
            not all(12 <= len(span) <= 280 for span in (first_quote, second_quote)) or
            any(replay.RISKY_TEXT.search(value) for value in (question, answer)) or
            not contains(text_key(question), text_key(source)) or
            contains(text_key(question), text_key(bridge)) or
            contains(text_key(question), text_key(answer)) or
            not contains(text_key(first_quote), text_key(bridge)) or
            not contains(text_key(second_quote), text_key(answer)) or
            any(contains(text_key(body), text_key(answer)) for _, body in first) or
            not has(first, source, first_quote) or not has(second, bridge, second_quote)):
        return None
    return {"qid": question_key(question), "question": question, "answer": answer, "family": family,
            "source": source, "bridge": bridge, "first": first, "second": second,
            "facts": [(title_key(source), text_key(first_quote)), (title_key(bridge), text_key(second_quote))]}


CAPITALIZED = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+(?:[A-Z][\w'’\-]*|of|the|and|de|von|van|in)){0,5}")
EDGE_WORDS = frozenset("the of and de von van in a an this that he she they it his her their its during from".split())
SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|[\r\n]+")


def text_spans(text):
    """Bound copied evidence by construction, while retaining contiguous source words."""
    spans = []
    for sentence in SENTENCE_BREAK.split(text):
        sentence = sentence.strip()
        if len(sentence) <= 280:
            if len(sentence) >= 12:
                spans.append(sentence)
            continue
        words = list(re.finditer(r"\S+", sentence))
        start = end = 0
        while start < len(words):
            end = max(start, end)
            while end < len(words) and words[end].end() - words[start].start() <= 280:
                end += 1
            if end > start:
                span = sentence[words[start].start():words[end - 1].end()]
                if len(span) >= 12:
                    spans.append(span)
            if end == len(words):
                break
            start = max(start + 1, end - 8)
    return list(dict.fromkeys(spans))


def bridge_options(source, body, blocked):
    spans, names = text_spans(body), {}
    source_text = text_key(source)
    for span in spans:
        for match in CAPITALIZED.finditer(span):
            words = match[0].split()
            while words and words[0].casefold() in EDGE_WORDS:
                words.pop(0)
            while words and words[-1].casefold() in EDGE_WORDS:
                words.pop()
            name = " ".join(words)
            key = title_key(name)
            normalized = text_key(name)
            if (not 4 <= len(name) <= 90 or key in blocked or normalized == source_text or
                    contains(source_text, normalized) or contains(normalized, source_text) or
                    not any(c.isalpha() for c in name)):
                continue
            priority = (-int(len(words) > 1), -sum(contains(text_key(s), normalized) for s in spans),
                        min(abs(len(words) - 2), 5), len(name), name)
            names.setdefault(key, (priority, name))
    return [(name, [span for span in spans if contains(text_key(span), text_key(name))][:4])
            for _, name in sorted(names.values())[:protocol.BRIDGE_LIMIT]]


def source_candidates(retriever, batch, blocked, funnel):
    first_results = retriever.search([source for _, source in batch])
    requests = []
    for (index, source), found in zip(batch, first_results, strict=True):
        first = clip([d for d in found if title_key(d[0]) == title_key(source)][:1])
        if not first:
            continue
        funnel["source_page_retrieved"] += 1
        options = bridge_options(source, first[0][1], blocked)
        funnel["sources_with_bridge_candidate"] += bool(options)
        for bridge, first_spans in options:
            family = protocol.FAMILIES[(index // 4) % len(protocol.FAMILIES)]
            requests.append((index, source, bridge, first, first_spans, family))
    retrieved = retriever.search([request[2] for request in requests])
    counts, tasks = Counter(), []
    for (index, source, bridge, first, first_spans, family), found in zip(requests, retrieved, strict=True):
        second = clip([doc for doc in found if title_key(doc[0]) == title_key(bridge)][:1])
        if not second:
            continue
        second_spans = text_spans(second[0][1])[:12]
        if second_spans and counts[index] < protocol.SYNTHESIS_PER_SOURCE:
            counts[index] += 1
            tasks.append((source, bridge, first, second, first_spans, second_spans, family))
    funnel["bridge_sources_retrieved"] += len(counts)
    funnel["bridge_paths_for_synthesis"] += len(tasks)
    return tasks, len(batch) + len(requests)


def synthesis_prompt(tok, source, bridge, first_spans, second_spans, family):
    return replay.render_ids(tok, [
        {"role": "system", "content": "Write one natural two-hop Wikipedia question, at most 220 characters. "
         "Use the exact source title in your question, describe rather than name the bridge, and do not include the "
         "answer in the question. Select a source span and a bridge span that jointly resolve the question. "
         "The answer is a short, exact substring of the selected bridge span, at most 72 characters. "
         f"Use a {family} relation if possible. Return one compact JSON object only with question, answer, "
         "and proof (two 1-based integer IDs, source first). Do not repeat the evidence text. "
         'Format: {"question":"...?", "answer":"...", "proof":[1,2]}'},
        {"role": "user", "content": json.dumps({"source": source, "bridge": bridge,
                                               "source_spans": dict(enumerate(first_spans, 1)),
                                               "bridge_spans": dict(enumerate(second_spans, 1))}, ensure_ascii=False)},
    ], tools=())


def decode_pair(raw, task, funnel):
    source, bridge, first, second, first_spans, second_spans, family = task
    value = answer_field(raw)
    if not value:
        funnel["unparseable_json"] += 1
        return None
    proof = value.get("proof")
    if (not isinstance(proof, list) or len(proof) != 2 or not all(type(i) is int for i in proof) or
            not 1 <= proof[0] <= len(first_spans) or not 1 <= proof[1] <= len(second_spans)):
        funnel["invalid_span_ids"] += 1
        return None
    value = {**value, "first_quote": first_spans[proof[0] - 1], "second_quote": second_spans[proof[1] - 1]}
    row = checked_pair(value, source, bridge, first, second, family)
    if row is None:
        funnel["semantic_or_evidence_rejected"] += 1
    return row


def prepare_shard(spec, locked, tok, model, device, generate, progress):
    retriever = wiki.Retriever(spec["retriever"])
    blocked = set(locked["blocked_train_pages"])
    forbidden_qids = {row["qid"] for rows in locked["splits"].values() for row in rows}
    start, stop = spec["source_range"]
    pages = list(enumerate(locked["titles"]))[start:stop][spec["shard"]::4]
    retained, queries, processed = [], 0, 0
    funnel = Counter()
    for batch in batches(pages):
        tasks, requested = source_candidates(retriever, batch, blocked, funnel)
        queries += requested
        valid = {}
        for candidates in batches(tasks):
            prompts = [synthesis_prompt(tok, source, bridge, first_spans, second_spans, family)
                       for source, bridge, _, _, first_spans, second_spans, family in candidates]
            generations = generate(model, tok, prompts, device, protocol.SYNTHESIS_TOKENS)
            for task, output in zip(candidates, generations, strict=True):
                funnel["length_stop"] += output["finish"] == "length"
                pair = decode_pair(output["text"], task, funnel)
                if pair and pair["qid"] not in forbidden_qids:
                    valid.setdefault((title_key(pair["source"]), title_key(pair["bridge"])), pair)
        results = retriever.search([row["question"] for row in valid.values()])
        queries += len(valid)
        for row, first in zip(valid.values(), results, strict=True):
            visible = [(title, body) for title, body in clip(first) if title_key(title) not in blocked]
            sees_bridge = has(visible, row["source"], row["facts"][0][1])
            complete = wiki.measure(visible, row["facts"])["all_missing_facts"]
            if sees_bridge and not complete:
                row["first"] = visible
            row["triage"] = "read" if complete else "search"
            row["question_first_witness"] = sees_bridge
            retained.append(row)
        processed += len(batch)
        progress("feedback_generating", processed, len(pages),
                 verified_pairs=len(retained), queries=queries)
    funnel["verified_before_encoding"] = len(retained)
    return {"pairs": retained, "queries": queries, "sources": len(pages), "funnel": dict(funnel)}


def merge_pairs(reports, locked):
    blocked = set(locked["blocked_train_pages"])
    ids, pairs, seen_pages = set(), [], set()
    candidates = [r for report in reports for r in report["pairs"]]
    bridge_frequency = Counter(title_key(row["bridge"]) for row in candidates)
    order = sorted(candidates, key=lambda row: (bridge_frequency[title_key(row["bridge"])], row["qid"]))
    for row in order:
        support = {title_key(row["source"]), title_key(row["bridge"])}
        observed = {title_key(title) for title, _ in [*row["first"], *row["second"]]}
        if row["qid"] not in ids and not (support & seen_pages or observed & blocked):
            pairs.append(row)
            ids.add(row["qid"])
            seen_pages.update(support)
    return pairs
