"""Test evidence presentation while holding actual Wikipedia returns and the reader fixed."""

import gzip
import html
import json
import re
import time
import unicodedata
from collections import Counter

from work3 import wiki_rsi_gap_follow_up as gap
from work3 import wiki_rsi_natural as natural
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_two_query_live as live
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import sha256, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_audit import support_facts
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_reader_evidence_v1"
SUMMARY = "wiki_rsi_w7_reader_summary.json"
SOURCE_ARM = live.POSITIVE[0]
ARMS = ("full", "compact", "observed_support")
WORD = re.compile(r"[^\W_]+", re.UNICODE)
WINDOW_WORDS = 48
MAX_WINDOWS = 8


def excerpt(body, sentence, surrounding=12):
    actual = unicodedata.normalize("NFKC", html.unescape(body))
    matches = list(WORD.finditer(actual))
    tokens = [item[0].casefold() for item in matches]
    needle = sentence.split()
    for start in range(len(tokens) - len(needle) + 1):
        if needle and tokens[start:start + len(needle)] == needle:
            left = matches[max(0, start - surrounding)].start()
            right = matches[min(len(matches) - 1, start + len(needle) + surrounding - 1)].end()
            return actual[left:right]
    return None


def observed_support(docs, facts):
    witness = []
    for wanted, sentence in sorted(facts):
        candidates = [(title, span) for title, body in docs
                      if title_key(title) == wanted and (span := excerpt(body, sentence))]
        if candidates:
            witness.append(min(candidates, key=lambda doc: len(doc[1])))
    return list(dict.fromkeys(witness))


def compact_evidence(question, follow_up, docs):
    terms = set(natural.words(question)) | set(natural.words(follow_up or ""))
    windows = []
    for title, body in docs:
        tokens = list(WORD.finditer(body))
        for start in range(0, len(tokens), WINDOW_WORDS // 2):
            last = min(start + WINDOW_WORDS, len(tokens))
            window = body[tokens[start].start():tokens[last - 1].end()]
            words = set(natural.words(f"{title} {window}"))
            windows.append((title, window, terms & words))
            if last == len(tokens):
                break
    frequency = Counter(term for _, _, overlap in windows for term in overlap)
    ranked = sorted(windows, key=lambda item: -sum(1 / frequency[word] for word in sorted(item[2])))
    chosen, titles = [], set()
    for title, body, _ in ranked:
        key = title_key(title)
        if key not in titles:
            chosen.append((title, body))
            titles.add(key)
    chosen.extend((title, body) for title, body, _ in ranked)
    return list(dict.fromkeys(chosen))[:MAX_WINDOWS]


def inputs(run_dir, references):
    prepared, ready, recorded = gap.prior_cases(run_dir)
    latest = replay.read_json(run_dir / gap.SUMMARY)
    if (latest["protocol"] != gap.PROTOCOL or latest["source_cases_sha256"] != sha256(run_dir / "w7_live_cases.json")
            or replay.read_json(run_dir / natural.PREP_SUMMARY)["inputs"]["references_sha256"] != sha256(references)):
        raise ValueError("Use the completed W7-gap run and original references for oracle scoring")
    with gzip.open(references, "rt", encoding="utf-8") as stream:
        refs = json.load(stream)["datasets"]
    facts = [support_facts(refs[row["dataset"]]["items"][row["qid"]]) for row in prepared["cases"]]
    if any(not group or any(not phrase for _, phrase in group) for group in facts):
        raise ValueError("Missing the complete reference support inventory for a W7b case")
    return prepared["cases"], ready, recorded, facts


def evaluate(cases, recorded, facts, tok, model, device, progress, generate):
    rows = []
    for offset in range(0, len(cases), live.BATCH):
        batch = list(zip(cases[offset:offset + live.BATCH], recorded[offset:offset + live.BATCH],
                         facts[offset:offset + live.BATCH], strict=True))
        items = []
        for case, old, required in batch:
            source = old["arms"][SOURCE_ARM]
            docs = [*case["visible"], *old["first_passages"], *source["second_passages"]]
            question = case["question"]
            packed = {"full": live.pack(tok, question, docs, live.answer_prompt, live.ANSWER_TOKENS)}
            actual = packed["full"][1]
            if wiki.measure(actual, case["missing_facts"])["all_missing_facts"] != source["evidence_complete"]:
                raise ValueError("The original R3 answer context no longer reproduces")
            for arm, visible in (("compact", compact_evidence(question, source["second_query"], actual)),
                                 ("observed_support", observed_support(actual, required))):
                packed[arm] = live.pack(tok, question, visible, live.answer_prompt, live.ANSWER_TOKENS)
            items.append({"dataset": case["dataset"], "qid": case["qid"], "cached_correct": source["correct"],
                          "packed": packed, "arms": {}})
        for arm in ARMS:
            answers = generate(model, tok, [item["packed"][arm][0] for item in items], device, live.ANSWER_TOKENS)
            for item, (case, _, required), answer in zip(items, batch, answers, strict=True):
                prompt, supplied = item["packed"][arm]
                item["arms"][arm] = {**replay.score_generation(answer["text"], case["aliases"], answer["finish"]),
                    "original_gap_complete": wiki.measure(supplied, case["missing_facts"])["all_missing_facts"],
                    "all_reference_facts": wiki.measure(supplied, required)["all_missing_facts"],
                    "prompt_tokens": len(prompt), "passages": supplied}
        for item in items:
            item.pop("packed")
        rows.extend(items)
        progress.update("running", len(rows), len(cases), detail="comparing_reader_evidence")
    return rows


def summarize_group(group):
    return {"n": len(group),
            "cached_full_correct": sum(row["cached_correct"] for row in group),
            "cached_full_decision_agreement": sum(row["cached_correct"] == row["arms"]["full"]["correct"] for row in group),
            "arms": {arm: {key: sum(row["arms"][arm][key] for row in group)
                           for key in ("correct", "valid", "original_gap_complete", "all_reference_facts", "prompt_tokens")}
                     for arm in ARMS},
            "compared_with_full": {arm: replay.paired([row["arms"] for row in group], arm, "full")
                                    for arm in ARMS[1:]}}


def summarize(rows):
    result = {}
    for dataset in SUPPORT_DATASETS:
        members = [row for row in rows if row["dataset"] == dataset]
        complete = [row for row in members if all(row["arms"][arm]["all_reference_facts"]
                                                 for arm in ("full", "observed_support"))]
        result[dataset] = {"all": summarize_group(members), "complete_reference_in_both": summarize_group(complete)}
    return result


def run(args, run_dir, progress):
    from work3 import wiki_rsi_action_model as model_tools
    from work3 import wiki_rsi_switch_train as training

    start = time.monotonic()
    cases, ready, recorded, facts = inputs(run_dir, args.references)
    replay.verify_checkpoint(args, ready)
    tok, model, device = training.load_solver(str(args.model_dir))
    rows = evaluate(cases, recorded, facts, tok, model, device, progress, model_tools.generate_queries)
    path = run_dir / "w7_reader_cases.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": rows})
    report = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT), "source_arm": SOURCE_ARM,
              "w7_gap_summary_sha256": sha256(run_dir / gap.SUMMARY), "output_sha256": sha256(path),
              "new_physical_queries": 0, "by_dataset": summarize(rows),
              "wall_seconds": round(time.monotonic() - start, 3)}
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done", len(rows), len(rows), detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0
