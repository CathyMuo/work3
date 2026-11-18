"""Measure one-query evidence acquisition from frozen W1 terminal states."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from work3 import p1_agent as solver
from work3 import wiki_rsi_replay as w1
from work3.evidence_ref import contains, question_key, sha256, text_key, title_key
from work3.p0_retrieval_diag import ROOT, SUPPORT_DATASETS, dataset_name, question_of, revision
from work3.wiki_rsi_audit import support_facts
from work3.wiki_rsi_progress import Progress, atomic_json, progress_root
from work3.wiki_rsi_references import WIKI_RSI_REFERENCES

PROTOCOL = "wiki_rsi_w2a_one_query_v1"
ARMS = ("original_question", "history_query", "support_title_probe")
QUERY_TOKENS = 64
TOPK = 3
ANCHORS_PER_DATASET = 4
QUERY_INSTRUCTION = (
    "Use the original question and search results above. Choose one natural-language Wikipedia "
    "search query to obtain a fact that would help answer the question and is still missing. "
    "Do not answer the question. Return exactly one query between <query> and </query> on one line."
)
QUERY_TAG = re.compile(r"<query>([^<>\r\n]{1,256})</query>")


def parse_query(text, finish):
    match = QUERY_TAG.fullmatch(text.strip()) if finish == "stop" else None
    return (match[1].strip() or None) if match else None


def normalized_docs(docs):
    return [(title_key(title), text_key(body)) for title, body in docs]


def measure(docs, missing):
    passages = normalized_docs(docs)
    matched = sum(any(title == key and contains(body, sentence) for key, body in passages)
                  for title, sentence in missing)
    return {"matched_facts": matched, "any_missing_fact": matched > 0,
            "all_missing_facts": matched == len(missing)}


class Retriever:
    def __init__(self, endpoint):
        url = urlsplit(endpoint)
        if (url.scheme, url.hostname, url.path) != ("http", "127.0.0.1", "/retrieve") or (
            not url.port or url.username or url.password or url.query or url.fragment
        ):
            raise ValueError("Use the original P0 loopback retriever URL, e.g. http://127.0.0.1:8000/retrieve")
        self.endpoint = endpoint
        self.client = build_opener(ProxyHandler({}))

    def search(self, queries):
        if not queries:
            return []
        payload = json.dumps({"queries": queries, "topk": TOPK, "return_scores": True}).encode()
        request = Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"})
        with self.client.open(request, timeout=45) as response:
            groups = json.load(response)["result"]
        if len(groups) != len(queries):
            raise ValueError("Retriever did not return one result group per query")
        parsed = []
        for group in groups:
            if not isinstance(group, list) or len(group) > TOPK:
                raise ValueError("Unexpected number of retriever results")
            docs = []
            for hit in group:
                content = hit["document"]["contents"]
                title, _, body = content.partition("\n")
                if not title:
                    raise ValueError("Retriever returned a passage without a title")
                docs.append((title, body))
            parsed.append(docs)
        return parsed


def check_anchors(retriever, anchors):
    observed = retriever.search([anchor["query"] for anchor in anchors])
    result = {ds: {"checked": 0, "exact_ordered": 0} for ds in SUPPORT_DATASETS}
    for anchor, docs in zip(anchors, observed, strict=True):
        row = result[anchor["dataset"]]
        row["checked"] += 1
        row["exact_ordered"] += [list(doc) for doc in normalized_docs(docs)] == anchor["expected"]
    ready = all(row["checked"] == ANCHORS_PER_DATASET and row["exact_ordered"] >= 3
                for row in result.values())
    return ready, result


def w1_records(run_dir):
    prepared, cases, _ = w1.verified_preparation(run_dir)
    if not prepared["ready_for_full_replay"]:
        raise ValueError("W2a requires a qualified W1 preparation")
    summary = w1.read_json(run_dir / "wiki_rsi_replay_summary.json")
    path = run_dir / "replay_cases.json"
    cases_file = w1.read_json(path)
    if (summary["protocol"] != w1.PROTOCOL or summary["mode"] != "replay"
            or summary["prepared_sha256"] != prepared["prepared_sha256"]
            or summary["prepare_summary_sha256"] != sha256(run_dir / "wiki_rsi_prepare_summary.json")
            or summary["completion_jobs"] != sum(len(case["prompts"]) for case in cases)
            or cases_file["protocol"] != w1.PROTOCOL):
        raise ValueError("W1 replay does not agree with the frozen preparation")
    results = {(row["dataset"], row["qid"]): row for row in cases_file["records"]}
    if len(results) != len(cases) or len(cases_file["records"]) != len(cases):
        raise ValueError("W1 replay records are incomplete or duplicated")

    rescued = set()
    for case in cases:
        key = case["dataset"], case["qid"]
        row = results.get(key)
        if row is None or row["cohort"] != case["cohort"] or set(row["arms"]) != set(case["prompts"]):
            raise ValueError("W1 replay record differs from its prepared case")
        if case["cohort"] == "missing_with_source":
            arms = row["arms"]
            if (all(arms[arm]["valid"] for arm in w1.ARMS[1:]) and arms["evidence"]["correct"]
                    and not (arms["empty"]["correct"] or arms["irrelevant"]["correct"])):
                rescued.add(key)
    for ds in SUPPORT_DATASETS:
        if sum(key[0] == ds for key in rescued) != summary["by_dataset"][ds]["evidence_specific_rescues"]:
            raise ValueError("W1 rescue count does not agree with the replay summary")
    return prepared, [case for case in cases if case["cohort"] == "missing_with_source"], rescued


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    prepared, cases, rescued = w1_records(run_dir)
    dump, test = w1.input_files(args)
    manifest = run_dir / "w0/wiki_rsi_manifest.jsonl"
    inputs = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
              "references_sha256": sha256(args.references), "manifest_sha256": sha256(manifest)}
    if any(prepared["source"].get(key) != digest for key, digest in inputs.items()):
        raise ValueError("Use the original P0/W1 inputs verified in the preparation summary")
    w1.verify_checkpoint(args, prepared)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    coords = {(row["dataset"], row["question_key"]): row["dump_line"]
              for _, row in w1.read_lines(manifest)}
    wanted = {(row["dataset"], row["qid"]) for row in cases}
    prompts = w1.initial_messages(test, wanted)
    lines = {coords[key] for key in wanted}
    target = {number: row for number, row in w1.read_lines(dump) if number in lines}
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = json.load(stream)["datasets"]

    records, anchors = [], []
    used_anchors = {ds: set() for ds in SUPPORT_DATASETS}
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    for number, case in enumerate(cases, 1):
        ds, qid = key = case["dataset"], case["qid"]
        stats = counts[ds]
        stats["w1_missing_cases"] += 1
        stats["w1_rescues"] += key in rescued
        start = prompts[key]
        question = question_of(start[-1]["content"])
        row = target[coords[key]]
        if dataset_name(row["data_source"]) != ds or question_key(question) != qid:
            raise ValueError("Target question disagrees with W1 preparation")
        prefix, boundary, _, trace = w1.reconstruct(tok, start, row)
        if boundary != case["prompts"]["original_replay"]:
            raise ValueError("Original history token IDs differ from frozen W1")
        queries = [query for message in prefix for call in message.get("tool_calls") or []
                   for query in call["function"]["arguments"]["query_list"]]
        for query, group in zip(queries, trace.queries, strict=True):
            if len(used_anchors[ds]) == ANCHORS_PER_DATASET:
                break
            if len(group) == TOPK and query not in used_anchors[ds]:
                used_anchors[ds].add(query)
                anchors.append({"dataset": ds, "query": query, "expected": group})
        if sum(message["role"] == "assistant" for message in prefix) + 2 > solver.MAX_TURNS:
            stats["excluded_turn_budget"] += 1
            continue
        proposal = w1.render_ids(tok, prefix + [solver.message("user", QUERY_INSTRUCTION)])
        if len(proposal) + QUERY_TOKENS + 1 > solver.MAX_MODEL_LEN:
            stats["excluded_proposal_context"] += 1
            continue
        reference = refs[ds]["items"][qid]
        visible = [doc for group in trace.queries for doc in group]
        missing = sorted((title, sentence) for title, sentence in support_facts(reference)
                         if not any(key == title and contains(body, sentence) for key, body in visible))
        if not missing:
            raise ValueError("Prepared missing case has no reproducible evidence gap")
        titles = Counter(title for title, _ in missing)
        title = min(titles, key=lambda key: (-titles[key], key))
        records.append({"dataset": ds, "qid": qid, "question": question, "missing_facts": missing,
                        "proposal": proposal, "probe_title": title, "w1_rescued": key in rescued,
                        "earliest_source_iteration": case["earliest_source_iteration"]})
        stats["eligible"] += 1
        stats["eligible_w1_rescues"] += key in rescued
        if number % 10 == 0:
            progress.update("running", number, len(cases), detail="checking_history")
    if not all(counts[ds]["eligible"] for ds in SUPPORT_DATASETS):
        raise ValueError("W2a found no eligible one-query states in one of the datasets")
    path = run_dir / "w2a_prepared.json"
    atomic_json(path, {"protocol": PROTOCOL, "records": records, "anchors": anchors})
    report = {"protocol": PROTOCOL, "repo_revision": revision(ROOT), "inputs": inputs,
              "w1_prepared_sha256": prepared["prepared_sha256"],
              "w1_cases_sha256": sha256(run_dir / "replay_cases.json"),
              "w2a_prepared_sha256": sha256(path), "by_dataset": counts}
    atomic_json(run_dir / "wiki_rsi_w2a_prepare_summary.json", report)
    progress.update("done", len(cases), len(cases), detail="ready",
                    metrics={ds: counts[ds]["eligible"] for ds in SUPPORT_DATASETS})
    print(json.dumps(report, indent=2), flush=True)
    return 0


def coverage_summary(records):
    groups = {}
    for ds in SUPPORT_DATASETS:
        dataset = [row for row in records if row["dataset"] == ds]
        scopes = {"all": dataset, "w1_evidence_specific_rescue": [row for row in dataset if row["w1_rescued"]],
                  "iter3_only": [row for row in dataset if row["earliest_source_iteration"] == 3],
                  "earlier_cache": [row for row in dataset if row["earliest_source_iteration"] != 3]}
        groups[ds] = {}
        for name, rows in scopes.items():
            pairs = [{arm: {"correct": int(row["arms"][arm]["all_missing_facts"])} for arm in ARMS}
                     for row in rows]
            groups[ds][name] = {
                "n": len(rows),
                "arms": {arm: {
                    "executed": sum(row["arms"][arm]["executed"] for row in rows),
                    "any_missing_fact": sum(row["arms"][arm]["any_missing_fact"] for row in rows),
                    "all_missing_facts": sum(row["arms"][arm]["all_missing_facts"] for row in rows),
                    "matched_facts": sum(row["arms"][arm]["matched_facts"] for row in rows),
                } for arm in ARMS},
                "history_vs_original_all_missing": w1.paired(pairs, "history_query", "original_question"),
            }
    return groups


def run(args, run_dir, progress):
    path = run_dir / "w2a_prepared.json"
    report_path = run_dir / "wiki_rsi_w2a_prepare_summary.json"
    report = w1.read_json(report_path)
    if report["protocol"] != PROTOCOL or report["w2a_prepared_sha256"] != sha256(path):
        raise ValueError("W2a input does not match its preparation summary")
    data = w1.read_json(path)
    cases = data["records"]
    profile = w1.runtime_config(args)
    prepared = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    if prepared["prepared_sha256"] != report["w1_prepared_sha256"]:
        raise ValueError("W2a and W1 refer to different frozen case sets")
    w1.verify_checkpoint(args, prepared)
    retriever = Retriever(args.retriever)
    progress.update("running", 0, len(cases), detail="checking_retriever")
    passed, anchors = check_anchors(retriever, data["anchors"])
    parity = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "topk": TOPK,
              "by_dataset": anchors, "passed": passed}
    atomic_json(run_dir / "wiki_rsi_w2a_retriever_check.json", parity)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(parity, indent=2), flush=True)
        return 2

    from vllm import LLM, SamplingParams

    progress.update("running", 0, len(cases), detail="loading_solver")
    model = LLM(model=str(args.model_dir), tensor_parallel_size=args.tp, max_model_len=solver.MAX_MODEL_LEN,
                dtype=profile["dtype"], seed=profile["seed"], gpu_memory_utilization=args.gpu_memory,
                enforce_eager=True, generation_config="vllm")
    params = SamplingParams(temperature=0, max_tokens=QUERY_TOKENS, skip_special_tokens=True)
    records = []
    generated_tokens = 0
    batch_size = 8
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset + batch_size]
        generations = model.generate([{"prompt_token_ids": case["proposal"]} for case in batch],
                                     sampling_params=params, use_tqdm=False)
        jobs, queries, arms = [], [], [{} for _ in batch]
        for idx, (case, output) in enumerate(zip(batch, generations, strict=True)):
            if list(output.prompt_token_ids) != case["proposal"] or len(output.outputs) != 1:
                raise RuntimeError("vLLM changed a query proposal prompt or completion count")
            generation = output.outputs[0]
            if len(generation.token_ids) > QUERY_TOKENS or generation.finish_reason not in {"stop", "length"}:
                raise RuntimeError("vLLM did not respect the query proposal budget")
            generated_tokens += len(generation.token_ids)
            query = parse_query(generation.text, generation.finish_reason)
            alternatives = (case["question"], query, case["probe_title"])
            for arm, text in zip(ARMS, alternatives, strict=True):
                arms[idx][arm] = {"query": text, "executed": bool(text), "passages": [],
                                  **measure([], case["missing_facts"])}
                if text:
                    jobs.append((idx, arm))
                    queries.append(text)
        for (idx, arm), docs in zip(jobs, retriever.search(queries), strict=True):
            arms[idx][arm].update(passages=docs, **measure(docs, batch[idx]["missing_facts"]))
        for case, results in zip(batch, arms, strict=True):
            records.append({key: case[key] for key in ("dataset", "qid", "w1_rescued",
                                                       "earliest_source_iteration")} | {"arms": results})
        progress.update("running", len(records), len(cases), detail="querying_wikipedia",
                        dataset=batch[-1]["dataset"],
                        metrics={"generated_tokens": generated_tokens})
    atomic_json(run_dir / "w2a_cases.json", {"protocol": PROTOCOL, "records": records})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(ROOT), "preparation_sha256": sha256(report_path),
               "retriever": parity, "runtime": {**profile, "max_new_tokens": QUERY_TOKENS},
               "generated_tokens": generated_tokens,
               "executed_queries": sum(arm["executed"] for row in records for arm in row["arms"].values()),
               "by_dataset": coverage_summary(records)}
    atomic_json(run_dir / "wiki_rsi_w2a_summary.json", summary)
    progress.update("done", len(records), len(cases), detail="completed",
                    metrics={"generated_tokens": generated_tokens, "queries": summary["executed_queries"]})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main(argv=None):
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "contract-prepare", "contract",
                                          "budget-prepare", "budget", "answer-prepare", "answer",
                                          "candidates-prepare", "candidates",
                                          "candidate-audit-prepare", "candidate-audit",
                                          "coverage-audit-prepare", "coverage-audit"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--drzero", type=Path, default=os.environ.get("DRZERO_DIR"))
    parser.add_argument("--eval-dump", type=Path, default=os.environ.get("P0_EVAL_DUMP"))
    parser.add_argument("--test-parquet", type=Path, default=os.environ.get("P0_TEST_PARQUET"))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("P0_ITER3_MODEL"))
    parser.add_argument("--references", type=Path, default=WIKI_RSI_REFERENCES)
    parser.add_argument("--corpus", type=Path, default=os.environ.get("WIKI_RSI_CORPUS"))
    parser.add_argument("--retriever", default=os.environ.get("WIKI_RSI_RETRIEVER"))
    parser.add_argument("--retriever-id", default=os.environ.get("WIKI_RSI_RETRIEVER_ID"))
    parser.add_argument("--expect-vllm", default=os.environ.get("WIKI_RSI_EXPECT_VLLM"))
    parser.add_argument("--gpu-memory", type=float, default=0.70)
    parser.add_argument("--native-gpus", help="optional additional GPU indices for inference batches")
    parser.set_defaults(tp=1)
    args = parser.parse_args(argv)
    if args.native_gpus is not None and args.phase not in {"contract", "budget", "answer"}:
        parser.error("--native-gpus applies to contract, budget and answer")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run):
        parser.error("--run requires the original W1 run name")
    cpu_stage = args.phase == "prepare" or args.phase.endswith("-prepare")
    no_model = {"candidates", "candidate-audit", "candidate-audit-prepare",
                "coverage-audit", "coverage-audit-prepare"}
    if args.phase not in no_model and not args.model_dir:
        parser.error("Set P0_ITER3_MODEL to the original P0 model")
    if cpu_stage and not args.drzero:
        parser.error("Set DRZERO_DIR to the original Dr. Zero checkout")
    if not cpu_stage and args.phase != "answer" and (not args.retriever or not args.retriever_id):
        parser.error("Set WIKI_RSI_RETRIEVER and WIKI_RSI_RETRIEVER_ID for the original P0 service")
    if not 0 < args.gpu_memory < 1:
        parser.error("--gpu-memory must be between zero and one")
    run_dir = progress_root() / args.run
    if not run_dir.is_dir():
        parser.error("Use the existing W1 run directory")
    if args.phase.startswith("coverage-audit"):
        from work3 import wiki_rsi_coverage_audit as coverage

        phase, action = (("w3_coverage_audit_prepare", coverage.prepare) if cpu_stage
                         else ("w3_coverage_audit", coverage.run))
    elif args.phase.startswith("candidate-audit"):
        from work3 import wiki_rsi_candidate_audit as audit

        phase, action = (("w3_candidate_audit_prepare", audit.prepare) if cpu_stage
                         else ("w3_candidate_audit", audit.run))
    elif args.phase.startswith("candidates"):
        from work3 import wiki_rsi_candidates as candidates

        phase, action = (("w3_candidates_prepare", candidates.prepare) if cpu_stage
                         else ("w3_candidates", candidates.run))
    elif args.phase.startswith("answer"):
        from work3 import wiki_rsi_w2_answer as answer

        phase, action = ("w2_answer_prepare", answer.prepare) if cpu_stage else ("w2_answer", answer.run)
    elif args.phase.startswith("budget"):
        from work3 import wiki_rsi_w2_budget as budget

        phase, action = ("w2_budget_prepare", budget.prepare) if cpu_stage else ("w2_budget", budget.run)
    elif args.phase.startswith("contract"):
        from work3 import wiki_rsi_w2_contract as contract

        phase, action = ("w2_contract_prepare", contract.prepare) if cpu_stage else ("w2_contract", contract.run)
    else:
        phase, action = ("w2a_prepare", prepare) if cpu_stage else ("w2a", run)
    result = run_dir / f"wiki_rsi_{phase}_summary.json"
    with (run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another stage is active for this run")
        if result.exists():
            parser.error("This stage has already completed")
        progress = Progress(run_dir, phase)
        progress.update("running", detail="checking_inputs")
        try:
            with progress.heartbeat():
                return action(args, run_dir, progress)
        except (Exception, KeyboardInterrupt) as error:
            row = progress.data["phases"][phase]
            progress.update("failed", row["done"], row["total"], detail=type(error).__name__)
            raise


if __name__ == "__main__":
    sys.exit(main())
