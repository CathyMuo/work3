"""One frozen free-query check on the unchanged W5a natural states."""

import json
import os
import re
import time
from collections import Counter

from work3 import wiki_rsi_natural as w5a
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_decode import DecodeEngine
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w5b_natural_query_v1"
W5A_CASES = "w5_natural_cases.json"
W5A_SUMMARY = "wiki_rsi_w5_natural_summary.json"
PREPARED = "w5_natural_query_prepared.json"
PREP_SUMMARY = "wiki_rsi_w5_natural_query_prepare_summary.json"
MAX_NEW = 96
MIN_NET_WINS = 10
PROMPT = ("Propose one short Wikipedia search query to resolve a still-missing fact needed "
          "to answer the question. Use the question and earlier retrieved passages to identify "
          "a specific entity and relevant attribute or relation. Return only the query on one "
          "line. Do not answer the question.")
CONTROLS = (*w5a.FIXED_ARMS, "any_fixed", "any_visible_candidate", "known_title")


def parse_query(text, finish):
    if finish != "stop":
        return None, "length"
    text = text.strip()
    if not text:
        return None, "empty"
    if len(text.splitlines()) != 1:
        return None, "multiline"
    wrapper = re.fullmatch(r"<query>\s*([^<>]+)\s*</query>", text, flags=re.IGNORECASE)
    if wrapper:
        text = wrapper[1].strip()
    text = re.sub(r"^query:\s*", "", text, count=1, flags=re.IGNORECASE)
    if len(text) >= 2 and (text[0], text[-1]) in {('"', '"'), ("“", "”")}:
        text = text[1:-1].strip()
    if not text_key(text) or len(text) > 256 or any(mark in text for mark in ("<", ">", "`")):
        return None, "invalid"
    return text, "executed"


def inputs(run_dir):
    records = w1.read_json(run_dir / w5a.PREPARED)
    prep = w1.read_json(run_dir / w5a.PREP_SUMMARY)
    checks = w1.read_json(run_dir / W5A_CASES)
    summary = w1.read_json(run_dir / W5A_SUMMARY)
    if (any(obj["protocol"] != w5a.PROTOCOL for obj in (records, prep, checks, summary))
            or prep["prepared_sha256"] != sha256(run_dir / w5a.PREPARED)
            or summary["preparation_sha256"] != sha256(run_dir / w5a.PREP_SUMMARY)
            or not summary["retriever"]["passed"]
            or records["retriever_id"] != summary["retriever"]["retriever_id"]
            or summary["decision"] != "one_frozen_query_generation_check"):
        raise ValueError("Use the completed W5a run that qualified for the frozen generation check")
    states = records["records"]
    by_id = {(row["dataset"], row["qid"]): row for row in checks["records"]}
    state_ids = {(row["dataset"], row["qid"]) for row in states}
    if len(by_id) != len(checks["records"]) or len(states) != len(state_ids) or state_ids != by_id.keys():
        raise ValueError("W5a state membership changed")
    verified = []
    for case in states:
        old = by_id[case["dataset"], case["qid"]]
        returned = old["returned"]
        if set(returned) != set(w5a.case_queries(case)) or w5a.score_case(case, returned) != old:
            raise ValueError("Recomputed W5a query outcomes do not match the recorded outcomes")
        verified.append(old)
    if w5a.summarize(verified) != (summary["by_dataset"], summary["decision"]):
        raise ValueError("W5a summary does not match the recorded cases")
    return records, verified


def prompt_ids(tok, question, visible):
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nEarlier Wikipedia results "
         f"(title, passage, in observed order):\n{json.dumps(visible, ensure_ascii=False)}"},
    ]
    return w1.render_ids(tok, messages, tools=())


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    frozen, verified = inputs(run_dir)
    checkpoint = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, checkpoint)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    rows, counts, overflows = [], Counter(), Counter()
    cases = frozen["records"]
    for index, (case, baseline) in enumerate(zip(cases, verified, strict=True), 1):
        ds = case["dataset"]
        ids = prompt_ids(tok, case["question"], case["visible"])
        if not ids:
            raise ValueError("Natural query prompt rendered empty")
        overflow = len(ids) + MAX_NEW + 1 > w1.solver.MAX_MODEL_LEN
        counts[ds] += 1
        overflows[ds] += overflow
        probe_words = text_key(case["probe_title"])
        literal = any(contains(text_key(text), probe_words)
                      for text in [case["question"], *(value for doc in case["visible"] for value in doc)])
        rows.append({"dataset": ds, "qid": case["qid"], "ids": None if overflow else ids,
                     "probe_title_literal": literal,
                     "missing_facts": case["missing_facts"], "baseline": {key: baseline[key] for key in CONTROLS}})
        progress.update("running", index, len(cases), detail="preparing_natural_query")
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "records": rows, "anchors": frozen["anchors"],
                       "retriever_id": frozen["retriever_id"]})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT),
               "prepared_sha256": sha256(path), "model_sha256": checkpoint["model_sha256"],
               "w5a_prepared_sha256": sha256(run_dir / w5a.PREPARED),
               "w5a_cases_sha256": sha256(run_dir / W5A_CASES), "w5a_summary_sha256": sha256(run_dir / W5A_SUMMARY),
               "by_dataset": counts, "context_overflow": overflows,
               "planned_generations": len(rows) - sum(overflows.values()), "max_new_tokens": MAX_NEW}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(rows), len(cases), detail="ready")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def summarize(rows):
    by_dataset = {}
    for ds in SUPPORT_DATASETS:
        members = [row for row in rows if row["dataset"] == ds]
        pairs = [{"frozen": {"correct": int(row["all_missing_facts"])},
                  **{arm: {"correct": int(row["baseline"][arm])} for arm in CONTROLS}} for row in members]
        by_dataset[ds] = {
            "n": len(members), "status": Counter(row["status"] for row in members),
            "any_missing_fact": sum(row["any_missing_fact"] for row in members),
            "all_missing_facts": sum(row["all_missing_facts"] for row in members),
            "beyond_visible_candidates": sum(row["all_missing_facts"] and not row["baseline"]["any_visible_candidate"]
                                            for row in members),
            "known_title_remaining": sum(row["baseline"]["known_title"] and not row["all_missing_facts"]
                                         for row in members),
            "pairwise_all_missing": {arm: w1.paired(pairs, "frozen", arm) for arm in CONTROLS},
            "by_literal_title": {
                label: {"n": sum(row["probe_title_literal"] == literal for row in members),
                        "all_missing_facts": sum(row["all_missing_facts"] and row["probe_title_literal"] == literal
                                                 for row in members)}
                for label, literal in (("visible", True), ("not_visible", False))
            },
        }
    comparisons = [row["pairwise_all_missing"]["any_fixed"] for row in by_dataset.values()]
    passed = all(row["n"] >= w5a.MIN_ROWS and row["wins"] - row["losses"] >= MIN_NET_WINS for row in comparisons)
    return by_dataset, ("qualifies_for_independent_action_training_pilot" if passed else "close_current_training_route")


def run(args, run_dir, progress):
    path = run_dir / PREPARED
    data, ready = w1.read_json(path), w1.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or ready["protocol"] != PROTOCOL
            or ready["prepared_sha256"] != sha256(path) or data["retriever_id"] != args.retriever_id
            or ready["w5a_prepared_sha256"] != sha256(run_dir / w5a.PREPARED)
            or ready["w5a_cases_sha256"] != sha256(run_dir / W5A_CASES)
            or ready["w5a_summary_sha256"] != sha256(run_dir / W5A_SUMMARY)):
        raise ValueError("Use the matching W5b preparation and original W5a outputs")
    if not re.fullmatch(r"[0-9]+", os.environ.get("CUDA_VISIBLE_DEVICES", "")):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one idle physical GPU")
    cases = data["records"]
    if (Counter(case["dataset"] for case in cases) != ready["by_dataset"]
            or sum(case["ids"] is not None for case in cases) != ready["planned_generations"]
            or ready["max_new_tokens"] != MAX_NEW):
        raise ValueError("The W5b planned generation count changed")
    w1.verify_checkpoint(args, ready)
    profile = {**w1.runtime_config(args), "max_new_tokens": MAX_NEW}
    start = time.monotonic()
    retriever = w2.Retriever(args.retriever)
    progress.update("running", 0, len(cases), detail="checking_retriever")
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "retriever_id": args.retriever_id, "passed": passed, "by_dataset": stats}
    atomic_json(run_dir / "wiki_rsi_w5_natural_query_retriever_check.json", check)
    if not passed:
        progress.update("blocked", 0, len(cases), detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    progress.update("running", 0, len(cases), detail="loading_solver")
    engine = DecodeEngine(args.model_dir, profile) if ready["planned_generations"] else None
    loaded = time.monotonic()
    rows = []
    queries, generated = 0, 0
    for offset in range(0, len(cases), 8):
        batch = cases[offset:offset + 8]
        scheduled = [case for case in batch if case["ids"] is not None]
        outputs = engine.generate([case["ids"] for case in scheduled]) if scheduled else []
        decoded = {(case["dataset"], case["qid"]): out for case, out in zip(scheduled, outputs, strict=True)}
        parsed = []
        for case in batch:
            output = decoded.get((case["dataset"], case["qid"]))
            query, status = parse_query(output["text"], output["finish"]) if output else (None, "context_overflow")
            parsed.append((query, status, output))
        wanted = list(dict.fromkeys(query for query, _, _ in parsed if query))
        returned = dict(zip(wanted, retriever.search(wanted), strict=True))
        generated += len(outputs)
        queries += len(wanted)
        for case, (query, status, output) in zip(batch, parsed, strict=True):
            passages = returned[query] if query else []
            rows.append({key: case[key] for key in ("dataset", "qid", "baseline", "probe_title_literal")} |
                        {"query": query, "status": status, "output": output, "passages": passages} |
                        w2.measure(passages, case["missing_facts"]))
        progress.update("running", len(rows), len(cases), detail="generating_queries",
                        metrics={"completion_jobs": generated, "queries": queries})
    by_dataset, decision = summarize(rows)
    atomic_json(run_dir / "w5_natural_query_cases.json", {"protocol": PROTOCOL, "records": rows})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "runtime": profile, "retriever": check,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "completion_jobs": generated,
               "physical_queries_excluding_anchors": queries, "load_seconds": round(loaded - start, 3),
               "wall_seconds": round(time.monotonic() - start, 3), "by_dataset": by_dataset,
               "minimum_net_wins_over_any_fixed_per_dataset": MIN_NET_WINS, "decision": decision}
    atomic_json(run_dir / "wiki_rsi_w5_natural_query_summary.json", summary)
    progress.update("done", len(rows), len(cases), detail="completed",
                    metrics={"completion_jobs": generated, "queries": queries})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
