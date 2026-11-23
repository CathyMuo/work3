"""Freeze page-disjoint evaluation and build paired evidence-grounded supervision."""

import gzip
import hashlib
import json
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


def source_prompt(tok, title, docs, family):
    return replay.render_ids(tok, [
        {"role": "system", "content": "Identify a Wikipedia entity literally named in the supplied result. "
         f"Try to find a {family} relation that can support a two-hop question. "
         "Choose one distinct entity with a separate article. Return JSON {\"bridge\":\"exact text\"}."},
        {"role": "user", "content": json.dumps({"source": title, "actual_return": clip(docs)}, ensure_ascii=False)},
    ], tools=())


def synthesis_prompt(tok, source, bridge, first, second, family):
    return replay.render_ids(tok, [
        {"role": "system", "content": "Write one short natural two-hop question that explicitly names the source "
         "article's subject and needs the relation to the bridge and one fact about the bridge. "
         "Do not name the bridge or answer in the question. Use the second result to give a short extractive answer. "
         f"Use a {family} relation if possible. Return JSON with question, answer, first_quote, second_quote. "
         "Quote a contiguous proof span from each article verbatim. The first quote includes the bridge name "
         "and the second quote includes the answer."},
        {"role": "user", "content": json.dumps({"source": source, "bridge": bridge, "first_return": clip(first),
                                               "second_return": clip(second)}, ensure_ascii=False)},
    ], tools=())


def prepare_shard(spec, locked, tok, model, device, generate, progress):
    retriever = wiki.Retriever(spec["retriever"])
    blocked = set(locked["blocked_train_pages"])
    forbidden_qids = {row["qid"] for rows in locked["splits"].values() for row in rows}
    pages = list(enumerate(locked["titles"]))[spec["shard"]::4]
    retained, queries = [], 0
    for batch in batches(pages):
        first_results = retriever.search([source for _, source in batch])
        queries += len(batch)
        seeds = [(source, clip([d for d in docs if title_key(d[0]) == title_key(source)][:1]),
                  protocol.FAMILIES[(index // 4) % len(protocol.FAMILIES)])
                 for (index, source), docs in zip(batch, first_results, strict=True)]
        seeds = [row for row in seeds if row[1]]
        if seeds:
            prompts = [source_prompt(tok, *row) for row in seeds]
            suggestions = generate(model, tok, prompts, device, 64)
            bridges = []
            for (source, docs, family), output in zip(seeds, suggestions, strict=True):
                name = answer_field(output["text"]).get("bridge")
                if (isinstance(name, str) and 4 <= len(name) <= 90 and
                        title_key(name) not in blocked | {title_key(source)} and
                        contains(text_key(docs[0][1]), text_key(name)) and not replay.RISKY_TEXT.search(name)):
                    bridges.append((source, name.strip(), docs, family))
            found = retriever.search([bridge for _, bridge, _, _ in bridges])
            queries += len(bridges)
            tasks = [(source, bridge, first, clip([doc for doc in second if title_key(doc[0]) == title_key(bridge)][:1]),
                      family) for (source, bridge, first, family), second in zip(bridges, found, strict=True)]
            tasks = [row for row in tasks if row[3]]
            if tasks:
                prompts = [synthesis_prompt(tok, *row) for row in tasks]
                generations = generate(model, tok, prompts, device, 320)
                valid = [pair for row, output in zip(tasks, generations, strict=True)
                         if (pair := checked_pair(answer_field(output["text"]), *row)) and pair["qid"] not in forbidden_qids]
                results = retriever.search([row["question"] for row in valid])
                queries += len(valid)
                for row, first in zip(valid, results, strict=True):
                    visible = [(title, body) for title, body in clip(first) if title_key(title) not in blocked]
                    sees_bridge = has(visible, row["source"], row["facts"][0][1])
                    complete = wiki.measure(visible, row["facts"])["all_missing_facts"]
                    if sees_bridge and not complete:
                        row["first"] = visible
                    row["triage"] = "read" if complete else "search"
                    row["question_first_witness"] = sees_bridge
                    retained.append(row)
        progress("feedback_generating", batch[-1][0] // 4 + 1, len(pages),
                 verified_pairs=len(retained), queries=queries)
    return {"pairs": retained, "queries": queries, "sources": len(pages)}


def merge_pairs(reports, locked):
    blocked = set(locked["blocked_train_pages"])
    ids, pairs, seen_pages = set(), [], set()
    for row in sorted((r for report in reports for r in report["pairs"]), key=lambda r: r["qid"]):
        support = {title_key(row["source"]), title_key(row["bridge"])}
        observed = {title_key(title) for title, _ in [*row["first"], *row["second"]]}
        if row["qid"] not in ids and not (support & seen_pages or observed & blocked):
            pairs.append(row)
            ids.add(row["qid"])
            seen_pages.update(support)
    return pairs
