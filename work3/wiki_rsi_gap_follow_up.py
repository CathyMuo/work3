"""Compare a missing-fact query instruction with a matched direct-query instruction."""

import json
import re
import time
from collections import Counter

from work3 import wiki_rsi_natural_query as query_format
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_switch_transfer as transfer
from work3 import wiki_rsi_two_query_live as live
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import sha256, text_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_gap_follow_up_v1"
SUMMARY = "wiki_rsi_w7_gap_summary.json"
ARMS = ("direct", "missing_fact")
INSTRUCTIONS = {
    "direct": "In Focus, name the main topic of the question. In Query, propose the most useful next search.",
    "missing_fact": "In Focus, name one necessary fact still unsupported by the observed passages. "
                    "In Query, search for that missing fact using the most relevant entity and relation.",
}


def prompt_ids(tok, question, observed, arm):
    return replay.render_ids(tok, [
        {"role": "system", "content": "You are choosing the next Wikipedia search. " + INSTRUCTIONS[arm] +
         " Output exactly two lines, Focus: and Query:, with one short phrase on each line. "
         "Do not answer the question. Return a standalone search query."},
        {"role": "user", "content": f"Question: {question}\n\n"
         f"Observed passages (title, passage):\n{json.dumps(observed, ensure_ascii=False)}"},
    ], tools=())


def follow_up(output, first, fallback):
    lines = output["text"].strip().splitlines()
    match = re.fullmatch(r"Focus:\s*([^<>]{1,160})", lines[0], re.I) if len(lines) == 2 else None
    query = None
    if match and match[1].strip() and re.match(r"^Query:\s*\S", lines[1], re.I):
        query, _ = query_format.parse_query(lines[1], output["finish"])
    tried = text_key(first or "")
    if query and text_key(query) != tried:
        return query, "model"
    if fallback and text_key(fallback) != tried:
        return fallback, "rule_fallback"
    return None, "no_second"


def prior_cases(run_dir):
    prior_path = run_dir / transfer.PREP_SUMMARY
    prior, result = [replay.read_json(run_dir / name) for name in (transfer.PREP_SUMMARY, transfer.SUMMARY)]
    live_path, live_summary = run_dir / live.PREPARED, run_dir / live.PREP_SUMMARY
    prepared, ready, reported = [replay.read_json(path) for path in (live_path, live_summary, run_dir / live.SUMMARY)]
    rows_path = run_dir / "w7_live_cases.json"
    saved = replay.read_json(rows_path)
    if (prior["protocol"] != transfer.PROTOCOL or result["protocol"] != transfer.PROTOCOL
            or result["decision"] != "natural_replay_complete"
            or result["preparation_sha256"] != sha256(prior_path)
            or prior["w7b_cases_sha256"] != sha256(rows_path)
            or any(row["protocol"] != live.PROTOCOL for row in (prepared, ready, reported, saved))
            or ready["prepared_sha256"] != sha256(live_path)
            or reported["preparation_sha256"] != sha256(live_summary)):
        raise ValueError("Use the original W7b cache and completed evidence-switch comparison")
    cases, records = prepared["cases"], saved["records"]
    transfer.verify_natural_totals(records, reported["by_dataset_and_stratum"])
    for case, record in zip(cases, records, strict=True):
        if any(case[key] != record[key] for key in ("dataset", "qid", "title_stratum")):
            raise ValueError("Original W7b case ordering changed")
    return prepared, ready, records


def evaluate(cases, recorded, tok, model, retriever, device, progress, generate):
    rows, physical = [], Counter()
    for offset in range(0, len(cases), live.BATCH):
        batch, base = cases[offset:offset + live.BATCH], recorded[offset:offset + live.BATCH]
        histories = [[*case["visible"], *row["first_passages"]] for case, row in zip(batch, base, strict=True)]
        outcomes = [{arm: value for arm, value in row["arms"].items()} for row in base]
        for arm in ARMS:
            def render(tokenizer, question, history):
                return prompt_ids(tokenizer, question, history, arm)

            prompts = [live.pack(tok, case["question"], history, render, query_format.MAX_NEW)[0]
                       for case, history in zip(batch, histories, strict=True)]
            proposals = generate(model, tok, prompts, device, query_format.MAX_NEW)
            plans = [follow_up(item, row["first_query"], case["linked_context"])
                     for case, row, item in zip(batch, base, proposals, strict=True)]
            returned, calls = live.search_batch(retriever, [query for query, _ in plans])
            physical[arm] += calls
            final = [[*history, *result[1]] for history, result in zip(histories, returned, strict=True)]
            packed = [live.pack(tok, case["question"], docs, live.answer_prompt, live.ANSWER_TOKENS)
                      for case, docs in zip(batch, final, strict=True)]
            answers = generate(model, tok, [ids for ids, _ in packed], device, live.ANSWER_TOKENS)
            for index, (case, answer) in enumerate(zip(batch, answers, strict=True)):
                query, passage = returned[index]
                required = case["missing_facts"]
                outcomes[index][arm] = {**replay.score_generation(answer["text"], case["aliases"], answer["finish"]),
                    "evidence_complete": wiki.measure(packed[index][1], required)["all_missing_facts"],
                    "raw_evidence_complete": wiki.measure(final[index], required)["all_missing_facts"],
                    "second_query": query, "second_passages": passage, "query_source": plans[index][1],
                    "queries": int(bool(base[index]["first_query"])) + int(bool(query))}
        rows.extend({key: case[key] for key in ("dataset", "qid", "title_stratum")} | {"arms": arms}
                    for case, arms in zip(batch, outcomes, strict=True))
        progress.update("running", len(rows), len(cases), detail="evaluating_missing_fact_queries")
    return rows, physical


def summarize(rows):
    groups = {}
    for dataset in SUPPORT_DATASETS:
        group = [row for row in rows if row["dataset"] == dataset]
        matched = [row for row in group if all(row["arms"][arm]["second_query"] for arm in ARMS)]
        groups[dataset] = {"n": len(group), "matched_second_query_n": len(matched), "arms": {
            arm: {key: sum(row["arms"][arm][key] for row in group)
                  for key in ("correct", "evidence_complete", "raw_evidence_complete", "valid", "queries")} | {
                      "evidence_but_incorrect": sum(row["arms"][arm]["evidence_complete"] and not row["arms"][arm]["correct"]
                                                   for row in group)}
            for arm in (*live.ARMS, *ARMS)},
            "query_source": {arm: Counter(row["arms"][arm]["query_source"] for row in group) for arm in ARMS},
            "gap_vs_direct": {key: live.paired_measure(group, "missing_fact", "direct", key)
                              for key in ("correct", "evidence_complete")},
            "gap_vs_direct_same_executed_budget": {key: live.paired_measure(matched, "missing_fact", "direct", key)
                                                  for key in ("correct", "evidence_complete")},
            "cached_or_stop_answer_ceiling": sum(any(row["arms"][arm]["correct"] for arm in live.ARMS) for row in group),
            "all_queries_or_stop_answer_ceiling": sum(any(item["correct"] for item in row["arms"].values()) for row in group),
            "gap_unique_answer": sum(row["arms"]["missing_fact"]["correct"] and
                                     not any(row["arms"][arm]["correct"] for arm in (*live.ARMS, "direct"))
                                     for row in group)}
    return groups


def run(args, run_dir, progress):
    from work3 import wiki_rsi_action_model as model_tools
    from work3 import wiki_rsi_switch_train as training

    start = time.monotonic()
    prepared, ready, recorded = prior_cases(run_dir)
    if args.retriever_id != prepared["retriever_id"]:
        raise ValueError("Use the original P0 retriever")
    replay.verify_checkpoint(args, ready)
    retriever = wiki.Retriever(args.retriever)
    passed, stats = wiki.check_anchors(retriever, prepared["anchors"])
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        print(json.dumps({"passed": passed, "by_dataset": stats}, indent=2), flush=True)
        return 2
    tok, model, device = training.load_solver(str(args.model_dir))
    rows, calls = evaluate(prepared["cases"], recorded, tok, model, retriever, device, progress, model_tools.generate_queries)
    path = run_dir / "w7_gap_cases.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": rows})
    report = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT), "output_sha256": sha256(path),
              "source_cases_sha256": sha256(run_dir / "w7_live_cases.json"), "by_dataset": summarize(rows),
              "physical_queries_excluding_anchors": calls, "wall_seconds": round(time.monotonic() - start, 3)}
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done", len(rows), len(rows), detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0
