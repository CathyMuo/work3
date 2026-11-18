"""Replay answers using the top results of actual one-query W2a searches."""

import gzip
import json
import time
from collections import Counter

from work3 import p1_agent as solver
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3 import wiki_rsi_w2_contract as contract
from work3.evidence_ref import question_key, sha256, text_key, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, dataset_name, question_of, revision
from work3.wiki_rsi_decode import DecodeEngine, native_gpu_ids, native_replicas
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w2b_real_results_answer_v1"
ARMS = ("empty", "original_query", "title_query", "matched_irrelevant")


def ranked_prefix(tok, prefix, docs, empty):
    retained = []
    prompt = empty
    for doc in docs:
        if any(w1.RISKY_TEXT.search(value) for value in doc):
            break
        candidate = w1.with_passages(tok, prefix, retained + [doc])
        if len(candidate) + w1.MAX_NEW + 1 > solver.MAX_MODEL_LEN:
            break
        retained.append(doc)
        prompt = candidate
    return retained, prompt


def matched_control(tok, prefix, docs, prompt, empty, banned_titles, aliases, pool):
    if not docs:
        return None
    try:
        negative = w1.choose_negative(tok, pool, docs, banned_titles, aliases)
    except w1.Excluded:
        return None
    control = w1.with_passages(tok, prefix, negative)
    if (len(control) + w1.MAX_NEW + 1 > solver.MAX_MODEL_LEN
            or abs(len(prompt) - len(control)) > max(8, 0.1 * (len(prompt) - len(empty)))):
        return None
    return control


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    recap, _, w2a, retrieved = contract.prior_artifacts(run_dir)
    w1_prep, w1_cases, _ = w1.verified_preparation(run_dir)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    dump, test = w1.input_files(args)
    files = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
             "references_sha256": sha256(args.references), "manifest_sha256": sha256(manifest)}
    if (recap["w1_prepared_sha256"] != w1_prep["prepared_sha256"]
            or any(recap["inputs"].get(key) != value for key, value in files.items())):
        raise ValueError("Use the verified original W1 and W2a inputs")
    w1.verify_checkpoint(args, w1_prep)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    cases = {(row["dataset"], row["qid"]): row for row in w1_cases}
    coords = {(row["dataset"], row["question_key"]): row["dump_line"]
              for _, row in w1.read_lines(manifest)}
    original = w1.initial_messages(test, set(retrieved))
    rows, _, catalog = w1.catalog(dump, [{"dump_line": coords[key], "support_passages": []}
                                         for key in retrieved])
    pools = {}
    for ds, items in catalog.items():
        lengths = w1.doc_lengths(tok, list(items.values())) if items else []
        pools[ds] = sorted((length, title, body, doc) for length, ((title, body), doc)
                           in zip(lengths, items.items(), strict=True))
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = json.load(stream)["datasets"]

    records = []
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    for number, candidate in enumerate(w2a["records"], 1):
        ds, qid = key = candidate["dataset"], candidate["qid"]
        case = cases[key]
        row = rows[coords[key]]
        initial = original[key]
        if (dataset_name(row["data_source"]) != ds or case["cohort"] != "missing_with_source"
                or question_key(question_of(initial[-1]["content"])) != qid):
            raise ValueError("Original question identity changed")
        prefix, boundary, _, trace = w1.reconstruct(tok, initial, row)
        empty = w1.with_passages(tok, prefix, [])
        if empty != case["prompts"]["empty"] or boundary != case["prompts"]["original_replay"]:
            raise ValueError("Original W1 history or empty prompt did not reproduce")

        names = {"original_query": "original_question", "title_query": "support_title_probe"}
        prompts = {"empty": empty}
        coverage = {}
        source = retrieved[key]["arms"]
        required = candidate["missing_facts"]
        stats = counts[ds]
        stats["states"] += 1
        title_docs = []
        for name, recorded in names.items():
            actual = source[recorded]
            docs = actual["passages"]
            if not actual["executed"] or len(docs) > w2.TOPK or w2.measure(docs, required) != {
                field: actual[field] for field in ("matched_facts", "any_missing_fact", "all_missing_facts")
            }:
                raise ValueError("Stored W2a search results do not reproduce their evidence coverage")
            kept, prompts[name] = ranked_prefix(tok, prefix, docs, empty)
            stats[f"{name}_all_raw"] += actual["all_missing_facts"]
            coverage[name] = w2.measure(kept, required)
            stats[f"{name}_all_packed"] += coverage[name]["all_missing_facts"]
            stats[f"{name}_lost_documents"] += len(docs) - len(kept)
            if name == "title_query":
                title_docs = kept
        forbidden = set(refs[ds]["items"][qid]["titles"])
        forbidden.update(title for group in trace.queries for title, _ in group)
        forbidden.update(title_key(title) for title, _ in title_docs)
        neg = matched_control(tok, prefix, title_docs, prompts["title_query"], empty, forbidden,
                              case["aliases"], pools[ds])
        if neg is not None:
            prompts["matched_irrelevant"] = neg
            stats["matched_controls"] += 1
        else:
            stats["no_matched_control"] += 1
        records.append({"dataset": ds, "qid": qid, "w1_rescued": candidate["w1_rescued"],
                        "prompts": prompts, "coverage": coverage, "aliases": case["aliases"]})
        if number % 8 == 0:
            progress.update("running", number, len(w2a["records"]), detail="checking_history", dataset=ds)
    summary_path = run_dir / "wiki_rsi_replay_summary.json"
    runtime = w1.read_json(summary_path)["runtime"]
    path = run_dir / "w2_answer_prepared.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": records, "w1_runtime": runtime})
    result = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "by_dataset": counts,
              "w1_prepared_sha256": w1_prep["prepared_sha256"], "w2a_cases_sha256": sha256(run_dir / "w2a_cases.json"),
              "prepared_sha256": sha256(path),
              "planned_completions": sum(len(row["prompts"]) for row in records)}
    atomic_json(run_dir / "wiki_rsi_w2_answer_prepare_summary.json", result)
    progress.update("done", len(records), len(records), detail="ready",
                    metrics={ds: counts[ds]["states"] for ds in SUPPORT_DATASETS})
    print(json.dumps(result, indent=2), flush=True)
    return 0


def summarize(records):
    def pair(rows, treatment, control):
        complete = [row["arms"] for row in rows if treatment in row["arms"] and control in row["arms"]]
        valid = [arms for arms in complete if arms[treatment]["valid"] and arms[control]["valid"]]
        return {"strict_em": w1.paired(complete, treatment, control),
                "both_valid": w1.paired(valid, treatment, control)}

    result = {}
    for ds in SUPPORT_DATASETS:
        rows = [row for row in records if row["dataset"] == ds]
        rescued = [row for row in rows if row["w1_rescued"]]
        result[ds] = {
            "n": len(rows),
            "arms": {arm: {metric: sum(row["arms"][arm][metric] for row in rows if arm in row["arms"])
                           for metric in ("correct", "valid", "p0_em")} | {
                               "n": sum(arm in row["arms"] for row in rows)} for arm in ARMS},
            "title_vs_original_query": pair(rows, "title_query", "original_query"),
            "title_vs_empty": pair(rows, "title_query", "empty"),
            "title_vs_matched_irrelevant": pair(rows, "title_query", "matched_irrelevant"),
            "w1_rescues_title_vs_empty": pair(rescued, "title_query", "empty"),
            "w1_rescues_title_vs_irrelevant": pair(rescued, "title_query", "matched_irrelevant"),
            "packed_title_all_facts_vs_empty": pair(
                [row for row in rows if row["coverage"]["title_query"]["all_missing_facts"]],
                "title_query", "empty"),
        }
    return result


def run(args, run_dir, progress):
    start = time.monotonic()
    path = run_dir / "w2_answer_prepared.json"
    report_path = run_dir / "wiki_rsi_w2_answer_prepare_summary.json"
    report, data = w1.read_json(report_path), w1.read_json(path)
    prepared = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    if (data["protocol"] != PROTOCOL or report["protocol"] != PROTOCOL
            or report["prepared_sha256"] != sha256(path)
            or report["w2a_cases_sha256"] != sha256(run_dir / "w2a_cases.json")
            or report["w1_prepared_sha256"] != prepared["prepared_sha256"]):
        raise ValueError("Use the matching W2b preparation and original W1 model")
    profile = w1.runtime_config(args)
    if any(data["w1_runtime"].get(key) != value for key, value in profile.items()):
        raise ValueError("Keep the W1 replay GPU and frozen answer decoder settings")
    devices = native_gpu_ids(profile["cuda_visible_devices"], args.native_gpus)
    w1.verify_checkpoint(args, prepared)
    cases = data["records"]
    batches = [cases[i:i + 8] for i in range(0, len(cases), 8)]
    jobs = [[(case, arm) for case in batch for arm in ARMS if arm in case["prompts"]]
            for batch in batches]
    inputs = [[case["prompts"][arm] for case, arm in batch] for batch in jobs]
    if not batches or sum(map(len, jobs)) != report["planned_completions"]:
        raise ValueError("W2b planned completion count changed")
    local_indices = list(range(0, len(batches), len(devices) + 1))
    local_set = set(local_indices)
    remote_indices = [index for index in range(len(batches)) if index not in local_set]
    gpu_count = 1 + min(len(devices), len(remote_indices))
    progress.update("running", 0, len(cases), detail="loading_solver", metrics={"inference_gpus": gpu_count})
    decoded = {}
    with native_replicas(run_dir, devices, [inputs[i] for i in remote_indices], args.model_dir, profile,
                         log_stem="w2_answer_gpu") as replicas:
        loaded_at = time.monotonic()
        engine = DecodeEngine(args.model_dir, profile)
        loaded = round(time.monotonic() - loaded_at, 3)
        done = 0
        for index in local_indices:
            replicas.require_healthy()
            decoded[index] = engine.generate(inputs[index])
            done += len(batches[index])
            progress.update("running", done, len(cases), detail="generating",
                            dataset=batches[index][-1]["dataset"], metrics={"inference_gpus": gpu_count})
        if devices:
            progress.update("running", done, len(cases), detail="waiting_answer_workers",
                            metrics={"inference_gpus": gpu_count})
        extra, timings = replicas.collect()
        decoded.update({remote_indices[index]: outputs for index, outputs in extra.items()})

    records, tokens = [], 0
    for index, batch in enumerate(batches):
        answers = {}
        for (case, arm), output in zip(jobs[index], decoded[index], strict=True):
            key = case["dataset"], case["qid"]
            answers.setdefault(key, {})[arm] = {
                **w1.score_generation(output["text"], case["aliases"], output["finish"]), **output}
            tokens += output["tokens"]
        records.extend({**{field: case[field] for field in ("dataset", "qid", "w1_rescued", "coverage")},
                        "arms": answers[case["dataset"], case["qid"]]} for case in batch)
        progress.update("running", len(cases), len(cases), detail="scoring_answers",
                        dataset=batch[-1]["dataset"], metrics={"inference_gpus": gpu_count,
                                                              "generated_tokens": tokens})
    atomic_json(run_dir / "w2_answer_cases.json", {"protocol": PROTOCOL, "records": records})
    elapsed = round(time.monotonic() - start, 3)
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "runtime": profile,
               "preparation_sha256": sha256(report_path), "by_dataset": summarize(records),
               "completion_jobs": sum(map(len, jobs)), "inference_gpus": gpu_count,
               "timings": {"load_seconds": loaded, "wall_seconds": elapsed, "replicas": timings},
               "generated_tokens": tokens}
    atomic_json(run_dir / "wiki_rsi_w2_answer_summary.json", summary)
    progress.update("done", len(records), len(cases), detail="completed",
                    metrics={"inference_gpus": gpu_count, "generated_tokens": tokens,
                             "completion_jobs": summary["completion_jobs"], "wall_seconds": elapsed})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
