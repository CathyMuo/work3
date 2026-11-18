"""Separate the generation budget from available evidence cues on frozen W2 states."""

import gzip
import hashlib
import json
import time
from collections import Counter

from work3 import p1_agent as solver
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3 import wiki_rsi_w2_contract as contract
from work3.evidence_ref import contains, question_key, sha256, text_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, dataset_name, question_of, revision
from work3.wiki_rsi_decode import DecodeEngine, native_gpu_ids, native_replicas
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w2_budget_cue_v1"
LONG_BUDGET = 192
BASE_ARMS = ("native64", "native192")
HINT_ARMS = ("target_title192", "other_title192")
HINT = "\nOne candidate Wikipedia page title, which may or may not be useful: "


def matching_title(qid, target, candidates, forbidden, lengths, question, visible, aliases):
    normalized_aliases = {text_key(alias) for alias in aliases}
    choices = [
        title for title in candidates if title != target and title not in forbidden
        and abs(lengths[target] - lengths[title]) <= 2
        and contract.title_cue(question, visible, title) == "no_literal_cue"
        and not any(contains(text_key(title), answer) for answer in normalized_aliases)
    ]
    return sorted(choices, key=lambda title: (abs(lengths[target] - lengths[title]),
                                             hashlib.sha256(f"{qid}:{title}".encode()).digest()))


def prior_inputs(run_dir):
    prep_path = run_dir / "wiki_rsi_w2_contract_prepare_summary.json"
    path = run_dir / "w2_contract_prepared.json"
    cases_path = run_dir / "w2_contract_cases.json"
    prep, data = w1.read_json(prep_path), w1.read_json(path)
    summary = w1.read_json(run_dir / "wiki_rsi_w2_contract_summary.json")
    cases = w1.read_json(cases_path)
    if (any(row["protocol"] != contract.PROTOCOL for row in (prep, data, summary, cases))
            or prep["new_prepared_sha256"] != sha256(path)
            or summary["preparation_sha256"] != sha256(prep_path)
            or not summary["retriever"]["passed"]):
        raise ValueError("Use the completed W2 query-contract run with its original inputs")
    old = {(case["dataset"], case["qid"]): case for case in cases["records"]}
    keys = {(case["dataset"], case["qid"]) for case in data["records"]}
    if len(old) != len(cases["records"]) or len(keys) != len(data["records"]) or old.keys() != keys:
        raise ValueError("Contract outcomes are incomplete")
    for ds in SUPPORT_DATASETS:
        rows = [row for (name, _), row in old.items() if name == ds]
        published = summary["by_dataset"][ds]["all"]
        native = [row["arms"]["native_prompt"] for row in rows]
        if len(rows) != published["n"] or any(
            sum(arm[metric] for arm in native) != count
            for metric, count in published["arms"]["native_prompt"].items()
        ):
            raise ValueError("Contract per-case results disagree with the original summary")
    return data, old, summary


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    data, old, summary = prior_inputs(run_dir)
    older = w1.read_json(run_dir / "wiki_rsi_w2a_prepare_summary.json")
    prepared, cases, _ = w1.verified_preparation(run_dir)
    dump, test = w1.input_files(args)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    hashes = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
              "references_sha256": sha256(args.references), "manifest_sha256": sha256(manifest)}
    if (older["w1_prepared_sha256"] != prepared["prepared_sha256"]
            or any(older["inputs"].get(key) != digest for key, digest in hashes.items())):
        raise ValueError("Original P0 inputs differ from W2a")
    w1.verify_checkpoint(args, prepared)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    coords = {(row["dataset"], row["question_key"]): row["dump_line"] for _, row in w1.read_lines(manifest)}
    targets = {coords[key] for key in old}
    rows = {number: row for number, row in w1.read_lines(dump) if number in targets}
    initial = w1.initial_messages(test, set(old))
    gold = {(case["dataset"], case["qid"]): case["aliases"] for case in cases}
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = json.load(stream)["datasets"]
    titles = {ds: sorted({case["probe_title"] for case in data["records"] if case["dataset"] == ds})
              for ds in SUPPORT_DATASETS}
    lengths = {title: len(tok(title, add_special_tokens=False)["input_ids"])
               for title in set().union(*titles.values())}
    ready = []
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    for index, case in enumerate(data["records"], 1):
        ds, qid = key = case["dataset"], case["qid"]
        stats = counts[ds]
        stats["original_states"] += 1
        row = rows[coords[key]]
        start = initial[key]
        question = question_of(start[-1]["content"])
        if dataset_name(row["data_source"]) != ds or question_key(question) != qid:
            raise ValueError("P0 question identity changed")
        prefix, _, _, trace = w1.reconstruct(tok, start, row)
        native = w1.render_ids(tok, prefix + [solver.message("user", contract.NATIVE_INSTRUCTION)])
        if native != case["native"]:
            raise ValueError("Original native query prompt did not reproduce")
        if len(native) + LONG_BUDGET + 1 > solver.MAX_MODEL_LEN:
            stats["budget_context_overflow"] += 1
            continue
        visible = [doc for group in trace.queries for doc in group]
        target = case["probe_title"]
        alternatives = matching_title(qid, target, titles[ds], refs[ds]["items"][qid]["titles"],
                                      lengths, question, visible, gold[key])
        target_prompt = w1.render_ids(tok, prefix + [
            solver.message("user", contract.NATIVE_INSTRUCTION + HINT + target)])
        hints = {}
        if len(target_prompt) + LONG_BUDGET + 1 <= solver.MAX_MODEL_LEN:
            for alternative in alternatives:
                other = w1.render_ids(tok, prefix + [
                    solver.message("user", contract.NATIVE_INSTRUCTION + HINT + alternative)])
                if (abs(len(other) - len(target_prompt)) <= 2
                        and len(other) + LONG_BUDGET + 1 <= solver.MAX_MODEL_LEN):
                    hints = dict(zip(HINT_ARMS, (target_prompt, other)))
                    break
        stats["budget_pairs"] += 1
        stats["cue_pairs" if hints else "cue_pair_unavailable"] += 1
        ready.append({**{field: case[field] for field in (
            "dataset", "qid", "w1_rescued", "title_cue", "missing_facts", "title_probe_all_missing")},
            "native": native, "hints": hints, "old_native": old[key]["raw_native"]})
        if index % 8 == 0:
            progress.update("running", index, len(data["records"]), detail="checking_history", dataset=ds)
    if not all(counts[ds]["budget_pairs"] for ds in SUPPORT_DATASETS):
        raise ValueError("No comparable generation-budget cases in one dataset")
    path = run_dir / "w2_budget_prepared.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": ready, "anchors": data["anchors"],
                       "runtime": summary["runtime"], "retriever_id": summary["retriever"]["retriever_id"]})
    result = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "by_dataset": counts,
              "contract_cases_sha256": sha256(run_dir / "w2_contract_cases.json"),
              "prepared_sha256": sha256(path)}
    atomic_json(run_dir / "wiki_rsi_w2_budget_prepare_summary.json", result)
    progress.update("done", len(data["records"]), len(data["records"]), detail="ready",
                    metrics={ds: counts[ds]["budget_pairs"] for ds in SUPPORT_DATASETS})
    print(json.dumps(result, indent=2), flush=True)
    return 0


def summarize(records):
    def scope(rows, arms, contrasts):
        pairs = [{arm: {"correct": int(row["arms"][arm]["all_missing_facts"])} for arm in arms}
                 for row in rows]
        return {
            "n": len(rows),
            "arms": {arm: {key: sum(row["arms"][arm][key] for row in rows) for key in
                           ("executed", "any_missing_fact", "all_missing_facts", "tokens")} | {
                               "finish": dict(Counter(row["arms"][arm]["finish"] for row in rows))}
                     for arm in arms},
            "paired_all_missing": {f"{a}_vs_{b}": w1.paired(pairs, a, b) for a, b in contrasts},
        }

    result = {}
    for ds in SUPPORT_DATASETS:
        rows = [row for row in records if row["dataset"] == ds]
        rescued = [row for row in rows if row["w1_rescued"]]
        matched = [row for row in rows if set(HINT_ARMS) <= row["arms"].keys()]
        comparisons = [(HINT_ARMS[0], HINT_ARMS[1]), (HINT_ARMS[0], BASE_ARMS[1]),
                       (HINT_ARMS[1], BASE_ARMS[1])]
        result[ds] = {
            "budget": scope(rows, BASE_ARMS, [(BASE_ARMS[1], BASE_ARMS[0])]),
            "budget_on_w1_rescues": scope(rescued, BASE_ARMS, [(BASE_ARMS[1], BASE_ARMS[0])]),
            "old_native64_exact_output": sum(row["reproduced"] for row in rows),
            "cue": scope(matched, BASE_ARMS + HINT_ARMS, comparisons),
            "cue_by_literal_availability": {
                group: scope([row for row in matched if
                              (row["title_cue"] != "no_literal_cue") == visible],
                             BASE_ARMS + HINT_ARMS, comparisons)
                for group, visible in (("visible", True), ("not_literal", False))},
        }
    return result


def run(args, run_dir, progress):
    start = time.monotonic()
    path = run_dir / "w2_budget_prepared.json"
    report_path = run_dir / "wiki_rsi_w2_budget_prepare_summary.json"
    report, data = w1.read_json(report_path), w1.read_json(path)
    if (data["protocol"] != PROTOCOL or report["protocol"] != PROTOCOL
            or report["prepared_sha256"] != sha256(path)
            or report["contract_cases_sha256"] != sha256(run_dir / "w2_contract_cases.json")
            or data["retriever_id"] != args.retriever_id):
        raise ValueError("Budget run does not match the frozen inputs or original retriever")
    runtime = w1.runtime_config(args)
    original = {**runtime, "max_new_tokens": w2.QUERY_TOKENS}
    if any(data["runtime"].get(key) != value for key, value in original.items()):
        raise ValueError("Keep the original single GPU and decode settings for the token-budget pair")
    profile = {**runtime, "max_new_tokens": LONG_BUDGET}
    devices = native_gpu_ids(runtime["cuda_visible_devices"], args.native_gpus)
    w1.verify_checkpoint(args, w1.read_json(run_dir / "wiki_rsi_prepare_summary.json"))
    cases = data["records"]
    progress.update("running", 0, len(cases), detail="checking_retriever")
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    parity = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w2_budget_retriever_check.json", parity)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(parity, indent=2), flush=True)
        return 2

    batches = [cases[i:i + 8] for i in range(0, len(cases), 8)]
    hint_batches = [[case["hints"][arm] for case in batch if case["hints"] for arm in HINT_ARMS]
                    for batch in batches]
    assigned = [index for index, batch in enumerate(hint_batches) if batch]
    replicas_input = [hint_batches[index] for index in assigned]
    gpu_count = 1 + min(len(devices), len(replicas_input))
    tokens = queries = 0
    produced, rows = [], []
    progress.update("running", 0, len(cases), detail="loading_solver", metrics={"inference_gpus": gpu_count})
    with native_replicas(run_dir, devices, replicas_input, args.model_dir, profile,
                         log_stem="w2_budget_hint_gpu") as replicas:
        loaded_at = time.monotonic()
        engine = DecodeEngine(args.model_dir, profile)
        loaded = round(time.monotonic() - loaded_at, 3)
        hint_results = {}
        for index, batch in enumerate(batches):
            replicas.require_healthy()
            ids = [case["native"] for case in batch]
            short = engine.generate(ids, max_tokens=w2.QUERY_TOKENS)
            long = engine.generate(ids)
            produced.append((short, long))
            if not devices and hint_batches[index]:
                hint_results[index] = engine.generate(hint_batches[index])
            progress.update("running", 0, len(cases), detail="generating_queries", dataset=batch[-1]["dataset"],
                            metrics={"legacy_proposals": min((index + 1) * 8, len(cases)),
                                     "inference_gpus": gpu_count})
        replica_stats = {}
        if devices:
            progress.update("running", 0, len(cases), detail="waiting_native_workers",
                            metrics={"inference_gpus": gpu_count})
            decoded, replica_stats = replicas.collect()
            hint_results = {assigned[index]: completions for index, completions in decoded.items()}
    for index, batch in enumerate(batches):
        lookups = {}
        hints = iter(hint_results.get(index, ()))
        for case, short, long in zip(batch, *produced[index], strict=True):
            generations = dict(zip(BASE_ARMS, (short, long)))
            if case["hints"]:
                generations.update((arm, next(hints)) for arm in HINT_ARMS)
            arms = {}
            for arm, output in generations.items():
                query, category = contract.compatible_query(output["text"], output["finish"])
                item = {**output, "query": query, "category": category, "executed": bool(query),
                        "passages": [], **w2.measure([], case["missing_facts"])}
                arms[arm] = item
                tokens += output["tokens"]
                if query:
                    lookups.setdefault(query, []).append((item, case["missing_facts"]))
            fields = {key: case[key] for key in ("dataset", "qid", "w1_rescued", "title_cue")}
            rows.append({**fields, "arms": arms, "reproduced": short == case["old_native"]})
        for query, docs in zip(lookups, retriever.search(list(lookups)), strict=True):
            for item, missing in lookups[query]:
                item.update(passages=docs, **w2.measure(docs, missing))
        queries += len(lookups)
        progress.update("running", len(rows), len(cases), detail="querying_wikipedia",
                        dataset=batch[-1]["dataset"], metrics={"queries": queries, "generated_tokens": tokens,
                                                              "inference_gpus": gpu_count})
    atomic_json(run_dir / "w2_budget_cases.json", {"protocol": PROTOCOL, "records": rows})
    result = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "runtime": runtime,
              "generation_budgets": [w2.QUERY_TOKENS, LONG_BUDGET], "retriever": parity,
              "preparation_sha256": sha256(report_path), "by_dataset": summarize(rows),
              "timings": {"load_seconds": loaded, "wall_seconds": round(time.monotonic() - start, 3),
                          "hint_replicas": replica_stats},
              "inference_gpus": gpu_count, "physical_queries_excluding_anchors": queries,
              "generated_tokens": tokens}
    atomic_json(run_dir / "wiki_rsi_w2_budget_summary.json", result)
    progress.update("done", len(rows), len(cases), detail="completed",
                    metrics={"queries": queries, "generated_tokens": tokens, "inference_gpus": gpu_count,
                             "wall_seconds": result["timings"]["wall_seconds"]})
    print(json.dumps(result, indent=2), flush=True)
    return 0
