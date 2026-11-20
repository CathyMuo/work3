"""Pack complementary earlier and follow-up evidence under a shared token cap."""

import json
import math
import time
from collections import Counter
from dataclasses import dataclass

from work3 import wiki_rsi_natural as natural
from work3 import wiki_rsi_reader_evidence as reader
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_two_query_live as live
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_reader_bridge_v1"
SUMMARY = "wiki_rsi_w7_bridge_summary.json"
INPUT_CAP = 640
WORDS_PER_WINDOW = 64
PAIR_POOL = 12
FILL_POOL = 24
MIN_DEV_NET = 5
MODES = ("independent", "linked")
ARMS = (*reader.ARMS, *MODES)


@dataclass(frozen=True)
class Window:
    title: str
    body: str
    doc: int
    start: int
    end: int
    follow_up: bool
    words: frozenset[str]


def windows_from(docs, follow_up):
    windows = []
    for doc, (title, body) in enumerate(docs):
        tokens = list(reader.WORD.finditer(body))
        for start in range(0, len(tokens), WORDS_PER_WINDOW // 2):
            end = min(len(tokens), start + WORDS_PER_WINDOW)
            text = body[tokens[start].start():tokens[end - 1].end()]
            windows.append(Window(title, text, doc, start, end, (title, body) in follow_up,
                                  frozenset(natural.words(text))))
            if end == len(tokens):
                break
    return windows


def rank_windows(windows, question, query):
    topic = set(natural.words(question)) | set(natural.words(query or ""))
    frequency = Counter(term for window in windows for term in window.words)
    weights = {term: 1 / math.sqrt(count) for term, count in frequency.items()}
    scores = {window: sum(weights[word] for word in sorted(topic & window.words)) for window in windows}
    ranked = sorted(windows, key=scores.__getitem__, reverse=True)
    return ranked, scores, topic, weights


def connection(left, right, topic, weights):
    mentioned = (contains(text_key(left.body), text_key(right.title))
                 or contains(text_key(right.body), text_key(left.title)))
    shared = sorted((left.words & right.words) - topic)
    rare = sum(sorted((weights[word] for word in shared), reverse=True)[:4])
    return 2.0 * mentioned + 0.5 * rare


def pack(tok, question, query, docs, follow_up, mode):
    ranked, scores, topic, weights = rank_windows(windows_from(docs, follow_up), question, query)
    early = [window for window in ranked if not window.follow_up][:PAIR_POOL]
    late = [window for window in ranked if window.follow_up][:PAIR_POOL]
    pairs = [(left, right) for left in early for right in late if title_key(left.title) != title_key(right.title)]
    pairs.sort(key=lambda pair: sum(scores[window] for window in pair)
               + (connection(*pair, topic, weights) if mode == "linked" else 0), reverse=True)

    def render(windows):
        supplied = [(window.title, window.body) for window in sorted(windows, key=lambda item: (item.doc, item.start))]
        return live.answer_prompt(tok, question, supplied), supplied

    selected = []
    ids, supplied = render(selected)
    if not ids or len(ids) > INPUT_CAP:
        raise ValueError("The question alone exceeds the shared evidence packing budget")
    for pair in pairs:
        candidate, evidence = render(pair)
        if len(candidate) <= INPUT_CAP:
            selected, ids, supplied = list(pair), candidate, evidence
            break
    for window in ranked[:FILL_POOL]:
        if any(item.doc == window.doc and item.start < window.end and window.start < item.end
               for item in selected):
            continue
        candidate, evidence = render([*selected, window])
        if len(candidate) <= INPUT_CAP:
            selected.append(window)
            ids, supplied = candidate, evidence
    return ids, supplied


def evaluate(cases, retrieved, facts, previous, tok, model, device, progress, generate):
    batches = list(zip(cases, retrieved, facts, previous, strict=True))
    results = []
    for offset in range(0, len(batches), live.BATCH):
        batch = batches[offset:offset + live.BATCH]
        rows = []
        for case, old, required, cached in batch:
            if any(case[key] != cached[key] for key in ("dataset", "qid")):
                raise ValueError("Original reader evaluation case order changed")
            source = old["arms"][reader.SOURCE_ARM]
            early = {tuple(doc) for doc in (*case["visible"], *old["first_passages"])}
            late = {tuple(doc) for doc in source["second_passages"]} - early
            docs = cached["arms"]["full"]["passages"]
            if wiki.measure(docs, required)["all_missing_facts"] != cached["arms"]["full"]["all_reference_facts"]:
                raise ValueError("Original evidence coverage does not reproduce")
            packed = {mode: pack(tok, case["question"], source["second_query"], docs, late, mode) for mode in MODES}
            rows.append({"dataset": case["dataset"], "qid": case["qid"], "arms": dict(cached["arms"]), "packed": packed})
        for mode in MODES:
            outputs = generate(model, tok, [row["packed"][mode][0] for row in rows], device, live.ANSWER_TOKENS)
            for row, (case, _, required, _), output in zip(rows, batch, outputs, strict=True):
                ids, docs = row["packed"][mode]
                row["arms"][mode] = {**replay.score_generation(output["text"], case["aliases"], output["finish"]),
                    "original_gap_complete": wiki.measure(docs, case["missing_facts"])["all_missing_facts"],
                    "all_reference_facts": wiki.measure(docs, required)["all_missing_facts"],
                    "prompt_tokens": len(ids), "passages": docs}
        for row in rows:
            row.pop("packed")
        results.extend(rows)
        progress.update("running", len(results), len(batches), detail="comparing_linked_evidence")
    return results


def group_summary(rows):
    return {"n": len(rows),
            "arms": {arm: {name: sum(row["arms"][arm][name] for row in rows)
                           for name in ("correct", "valid", "original_gap_complete", "all_reference_facts",
                                        "prompt_tokens")} for arm in ARMS},
            "linked_vs": {baseline: replay.paired([row["arms"] for row in rows], "linked", baseline)
                          for baseline in ("independent", "full")}}


def summarize(rows):
    result = {}
    for dataset in SUPPORT_DATASETS:
        group = [row for row in rows if row["dataset"] == dataset]
        complete = [row for row in group if row["arms"]["full"]["all_reference_facts"]]
        result[dataset] = {"all": group_summary(group), "full_has_all_reference_facts": group_summary(complete)}
    return result


def verdict(report):
    target = report["2wikimultihopqa"]
    paired = target["all"]["linked_vs"]
    counts = target["full_has_all_reference_facts"]["arms"]
    if (all(value["wins"] - value["losses"] >= MIN_DEV_NET for value in paired.values())
            and counts["linked"]["all_reference_facts"] > counts["independent"]["all_reference_facts"]):
        return "eligible_for_unseen_2wiki_replication"
    return "no_go_for_linked_packing"


def run(args, run_dir, progress):
    from work3 import wiki_rsi_action_model as model_tools
    from work3 import wiki_rsi_switch_train as training

    start = time.monotonic()
    previous_path = run_dir / "w7_reader_cases.json"
    previous = replay.read_json(previous_path)
    receipt = replay.read_json(run_dir / reader.SUMMARY)
    if (previous["protocol"] != reader.PROTOCOL or receipt["protocol"] != reader.PROTOCOL
            or receipt["source_arm"] != reader.SOURCE_ARM or receipt["output_sha256"] != sha256(previous_path)
            or receipt["w7_gap_summary_sha256"] != sha256(run_dir / reader.gap.SUMMARY)):
        raise ValueError("Use the completed fixed-retrieval reader experiment")
    cases, ready, old, facts = reader.inputs(run_dir, args.references)
    replay.verify_checkpoint(args, ready)
    tok, model, device = training.load_solver(str(args.model_dir))
    rows = evaluate(cases, old, facts, previous["records"], tok, model, device, progress, model_tools.generate_queries)
    path = run_dir / "w7_bridge_cases.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": rows})
    groups = summarize(rows)
    report = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT),
              "source_reader_sha256": sha256(run_dir / reader.SUMMARY), "output_sha256": sha256(path),
              "new_physical_queries": 0, "input_token_cap": INPUT_CAP, "by_dataset": groups,
              "research_verdict": verdict(groups),
              "wall_seconds": round(time.monotonic() - start, 3)}
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done", len(rows), len(rows), detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0
