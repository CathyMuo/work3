"""Separate rejected serialization from query targeting on the frozen W2a cases."""

import json
import re
import time
from collections import Counter

from work3 import p1_agent as solver
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, question_key, sha256, text_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, dataset_name, question_of, revision
from work3.wiki_rsi_progress import atomic_json
from work3.wiki_rsi_decode import DecodeEngine, native_gpu_ids, native_replicas

PROTOCOL = "wiki_rsi_w2_query_contract_v1"
ARMS = ("strict_repeat", "same_bytes_compatible", "native_prompt")
NATIVE_INSTRUCTION = (
    "Use the original question and search results above. Choose one natural-language Wikipedia "
    "search query to obtain a fact that would help answer the question and is still missing. "
    'Do not answer the question. Use one search tool call with one query: '
    '<tool_call>{"name":"search","arguments":{"query_list":["your query"]}}</tool_call>'
)
THINK = re.compile(r"<think>((?:(?!</?think>).)*)</think>\s*", re.DOTALL)
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def compatible_query(text, finish):
    if finish != "stop":
        return None, "length" if finish == "length" else "other_finish"
    clean = text.strip()
    thought = THINK.match(clean)
    if thought is not None:
        clean = clean[thought.end():]
    query = w2.parse_query(clean, finish)
    if query is not None:
        return query, "query_after_think" if thought else "standalone_query"
    action = CALL.fullmatch(clean)
    if action is None:
        return None, "no_single_action"
    try:
        data = json.loads(action[1])
    except ValueError:
        return None, "invalid_tool_json"
    if (not isinstance(data, dict) or set(data) != {"name", "arguments"}
            or data["name"] != "search" or not isinstance(data["arguments"], dict)
            or set(data["arguments"]) != {"query_list"}):
        return None, "wrong_tool_schema"
    queries = data["arguments"]["query_list"]
    if not isinstance(queries, list) or len(queries) != 1 or not isinstance(queries[0], str):
        return None, "wrong_query_count_or_type"
    query = w2.parse_query(f"<query>{queries[0]}</query>", finish)
    return (query, "native_search") if query else (None, "invalid_query_text")


def title_cue(question, passages, title):
    needle = text_key(title)
    in_question = contains(text_key(question), needle)
    in_return = any(contains(text_key(key), needle) or contains(body, needle) for key, body in passages)
    if in_question and in_return:
        return "both"
    if in_question:
        return "question"
    return "returned_passage" if in_return else "no_literal_cue"


def prior_artifacts(run_dir):
    path = run_dir / "w2a_prepared.json"
    prep_path = run_dir / "wiki_rsi_w2a_prepare_summary.json"
    recap = w1.read_json(prep_path)
    summary = w1.read_json(run_dir / "wiki_rsi_w2a_summary.json")
    old_path = run_dir / "w2a_cases.json"
    packs, old_data = w1.read_json(path), w1.read_json(old_path)
    if (any(row["protocol"] != w2.PROTOCOL for row in (recap, summary, packs, old_data))
            or recap["w2a_prepared_sha256"] != sha256(path)
            or summary["preparation_sha256"] != sha256(prep_path)
            or not summary["retriever"]["passed"]):
        raise ValueError("Completed W2a does not match the frozen inputs")
    old = {(case["dataset"], case["qid"]): case for case in old_data["records"]}
    keys = {(case["dataset"], case["qid"]) for case in packs["records"]}
    if len(old) != len(old_data["records"]) or len(keys) != len(packs["records"]) or old.keys() != keys:
        raise ValueError("W2a case coverage does not match its frozen inputs")
    for ds in SUPPORT_DATASETS:
        sample = [case for (dataset, _), case in old.items() if dataset == ds]
        expected = summary["by_dataset"][ds]["all"]
        if (len(sample) != expected["n"] or any(
            sum(case["arms"][arm][metric] for case in sample) != value
            for arm, metrics in expected["arms"].items() for metric, value in metrics.items()
        )):
            raise ValueError("W2a per-case results disagree with the published run summary")
    return recap, summary, packs, old


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    recap, summary, packs, old = prior_artifacts(run_dir)
    prepared = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    dump, test = w1.input_files(args)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    assets = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
              "manifest_sha256": sha256(manifest)}
    if (prepared["prepared_sha256"] != recap["w1_prepared_sha256"]
            or any(recap["inputs"].get(key) != value for key, value in assets.items())):
        raise ValueError("Use the same original P0 inputs as the completed W2a")
    w1.verify_checkpoint(args, prepared)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    coords = {(case["dataset"], case["question_key"]): case["dump_line"]
              for _, case in w1.read_lines(manifest)}
    keys = set(old)
    initial = w1.initial_messages(test, keys)
    targets = {coords[key] for key in keys}
    rows = {number: row for number, row in w1.read_lines(dump) if number in targets}
    retained = []
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    for index, case in enumerate(packs["records"], 1):
        ds, qid = key = case["dataset"], case["qid"]
        row = rows[coords[key]]
        start = initial[key]
        question = question_of(start[-1]["content"])
        if question_key(question) != qid or dataset_name(row["data_source"]) != ds:
            raise ValueError("Case disagrees with the original P0 trace")
        prefix, _, _, trace = w1.reconstruct(tok, start, row)
        legacy = w1.render_ids(tok, prefix + [solver.message("user", w2.QUERY_INSTRUCTION)])
        if legacy != case["proposal"]:
            raise ValueError("Original W2a proposal token IDs did not reproduce")
        native = w1.render_ids(tok, prefix + [solver.message("user", NATIVE_INSTRUCTION)])
        counts[ds]["original_states"] += 1
        saved = old[key]["arms"]
        cue = title_cue(question, [doc for group in trace.queries for doc in group], case["probe_title"])
        counts[ds][f"cue:{cue}"] += 1
        if len(native) + w2.QUERY_TOKENS + 1 <= solver.MAX_MODEL_LEN:
            retained.append({**case, "native": native, "title_cue": cue,
                             "old_query": saved["history_query"]["query"],
                             "title_probe_all_missing": saved["support_title_probe"]["all_missing_facts"]})
            counts[ds]["retained"] += 1
        else:
            counts[ds]["native_context_overflow"] += 1
        if index % 10 == 0:
            progress.update("running", index, len(packs["records"]), detail="checking_history")
    if not all(counts[ds]["retained"] for ds in SUPPORT_DATASETS):
        raise ValueError("No comparable prompt states for one dataset")
    path = run_dir / "w2_contract_prepared.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": retained, "anchors": packs["anchors"],
                       "w2a_retriever_id": summary["retriever"]["retriever_id"]})
    result = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "by_dataset": counts,
              "w2a_prepared_sha256": recap["w2a_prepared_sha256"],
              "w2a_cases_sha256": sha256(run_dir / "w2a_cases.json"),
              "new_prepared_sha256": sha256(path)}
    atomic_json(run_dir / "wiki_rsi_w2_contract_prepare_summary.json", result)
    progress.update("done", len(packs["records"]), len(packs["records"]), detail="ready",
                    metrics={ds: counts[ds]["retained"] for ds in SUPPORT_DATASETS})
    print(json.dumps(result, indent=2), flush=True)
    return 0


def summarize(rows):
    def scope(sample):
        pairs = [{name: {"correct": int(case["arms"][name]["all_missing_facts"])} for name in ARMS}
                 for case in sample]
        return {
            "n": len(sample),
            "old_strict_query_reproduced": sum(row["arms"]["strict_repeat"]["query"] == row["old_query"]
                                               for row in sample),
            "old_title_probe_all_missing": sum(row["title_probe_all_missing"] for row in sample),
            "arms": {arm: {key: sum(row["arms"][arm][key] for row in sample)
                           for key in ("executed", "any_missing_fact", "all_missing_facts")} for arm in ARMS},
            "parse_only_all_missing": w1.paired(pairs, "same_bytes_compatible", "strict_repeat"),
            "native_vs_compatible_all_missing": w1.paired(pairs, "native_prompt", "same_bytes_compatible"),
            "legacy_output_types": dict(Counter(row["legacy_type"] for row in sample)),
            "native_output_types": dict(Counter(row["native_type"] for row in sample)),
        }
    report = {}
    for ds in SUPPORT_DATASETS:
        dataset = [row for row in rows if row["dataset"] == ds]
        report[ds] = {"all": scope(dataset),
                      "w1_rescued": scope([row for row in dataset if row["w1_rescued"]]),
                      "by_literal_cue": {category: scope([row for row in dataset if row["title_cue"] == category])
                                         for category in ("both", "question", "returned_passage", "no_literal_cue")}}
    return report


def decode_batches(args, run_dir, progress, batches, profile, devices):
    native_inputs = [[case["native"] for case in batch] for batch in batches]
    legacy, native = [], {}
    gpu_count = 1 + min(len(devices), len(batches))
    metrics = {"inference_gpus": gpu_count}
    total = sum(map(len, batches))
    progress.update("running", 0, total, detail="loading_solver", metrics=metrics)
    start = time.monotonic()
    with native_replicas(run_dir, devices, native_inputs, args.model_dir, profile) as replicas:
        load_start = time.monotonic()
        engine = DecodeEngine(args.model_dir, profile)
        load_seconds = time.monotonic() - load_start
        legacy_seconds = native_seconds = 0
        issued = 0
        for index, batch in enumerate(batches):
            replicas.require_healthy()
            before = time.monotonic()
            legacy.append(engine.generate([case["proposal"] for case in batch]))
            legacy_seconds += time.monotonic() - before
            if not devices:
                before = time.monotonic()
                native[index] = engine.generate(native_inputs[index])
                native_seconds += time.monotonic() - before
            issued += len(batch)
            progress.update("running", 0, total, detail="generating_queries",
                            dataset=batch[-1]["dataset"],
                            metrics={**metrics, "legacy_proposals": issued})
        native_reports = {}
        if devices:
            progress.update("running", 0, total, detail="waiting_native_workers",
                            metrics={**metrics, "legacy_proposals": issued})
            native, native_reports = replicas.collect()
    timings = {
        "legacy_model_load_seconds": round(load_seconds, 3),
        "legacy_generation_seconds": round(legacy_seconds, 3),
        "local_native_generation_seconds": round(native_seconds, 3),
        "native_replicas": native_reports,
        "model_load_and_generation_wall_seconds": round(time.monotonic() - start, 3),
    }
    return legacy, native, timings


def run(args, run_dir, progress):
    start = time.monotonic()
    path = run_dir / "w2_contract_prepared.json"
    report_path = run_dir / "wiki_rsi_w2_contract_prepare_summary.json"
    report, data = w1.read_json(report_path), w1.read_json(path)
    if (report["protocol"] != PROTOCOL or data["protocol"] != PROTOCOL
            or report["new_prepared_sha256"] != sha256(path)
            or report["w2a_cases_sha256"] != sha256(run_dir / "w2a_cases.json")
            or data["w2a_retriever_id"] != args.retriever_id):
        raise ValueError("Use the verified W2a case files and retriever identity")
    profile = w1.runtime_config(args)
    prior = w1.read_json(run_dir / "wiki_rsi_w2a_summary.json")["runtime"]
    compare = {**profile, "max_new_tokens": w2.QUERY_TOKENS}
    if any(prior.get(key) != value for key, value in compare.items()):
        raise ValueError("Use the same GPU and generation configuration as W2a for the legacy repeat")
    devices = native_gpu_ids(profile["cuda_visible_devices"], args.native_gpus)
    w1.verify_checkpoint(args, w1.read_json(run_dir / "wiki_rsi_prepare_summary.json"))
    cases = data["records"]
    progress.update("running", 0, len(cases), detail="checking_retriever")
    retriever = w2.Retriever(args.retriever)
    passed, anchors = w2.check_anchors(retriever, data["anchors"])
    parity = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": anchors}
    atomic_json(run_dir / "wiki_rsi_w2_contract_retriever_check.json", parity)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(parity, indent=2), flush=True)
        return 2

    batches = [cases[offset:offset + 8] for offset in range(0, len(cases), 8)]
    checked_at = time.monotonic()
    legacy_batches, native_batches, timings = decode_batches(
        args, run_dir, progress, batches, compare, devices)
    timings["input_and_anchor_checks_seconds"] = round(checked_at - start, 3)
    gpu_count = 1 + len(timings["native_replicas"])
    rows = []
    generated_tokens = 0
    physical_queries = 0
    retrieval_started = time.monotonic()
    for index, batch in enumerate(batches):
        lookups, chunk = {}, []
        for case, legacy, native in zip(batch, legacy_batches[index], native_batches[index], strict=True):
            generated_tokens += legacy["tokens"] + native["tokens"]
            strict = w2.parse_query(legacy["text"], legacy["finish"])
            compatible, legacy_type = compatible_query(legacy["text"], legacy["finish"])
            alternate, native_type = compatible_query(native["text"], native["finish"])
            arms = {}
            for arm, query in zip(ARMS, (strict, compatible, alternate), strict=True):
                item = {"query": query, "executed": bool(query), "passages": [],
                        **w2.measure([], case["missing_facts"])}
                arms[arm] = item
                if query:
                    lookups.setdefault(query, []).append((item, case["missing_facts"]))
            selection = {key: case[key] for key in ("dataset", "qid", "title_cue", "w1_rescued",
                                                     "old_query", "title_probe_all_missing")}
            chunk.append({**selection, "legacy_type": legacy_type, "native_type": native_type,
                          "raw_legacy": legacy, "raw_native": native, "arms": arms})
        for query, docs in zip(lookups, retriever.search(list(lookups)), strict=True):
            for item, missing in lookups[query]:
                item.update(passages=docs, **w2.measure(docs, missing))
        physical_queries += len(lookups)
        rows.extend(chunk)
        progress.update("running", len(rows), len(cases), detail="querying_wikipedia",
                        dataset=batch[-1]["dataset"],
                        metrics={"generated_tokens": generated_tokens, "queries": physical_queries,
                                 "inference_gpus": gpu_count})
    timings["retrieval_and_scoring_seconds"] = round(time.monotonic() - retrieval_started, 3)
    timings["wall_seconds"] = round(time.monotonic() - start, 3)
    atomic_json(run_dir / "w2_contract_cases.json", {"protocol": PROTOCOL, "records": rows})
    result = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "runtime": compare,
              "inference_gpus": gpu_count, "timings": timings,
              "retriever": parity, "physical_queries_excluding_anchors": physical_queries,
              "generated_tokens": generated_tokens, "preparation_sha256": sha256(report_path),
              "by_dataset": summarize(rows)}
    atomic_json(run_dir / "wiki_rsi_w2_contract_summary.json", result)
    progress.update("done", len(rows), len(cases), detail="completed",
                    metrics={"generated_tokens": generated_tokens, "queries": physical_queries,
                             "inference_gpus": gpu_count, "wall_seconds": timings["wall_seconds"]})
    print(json.dumps(result, indent=2), flush=True)
    return 0
