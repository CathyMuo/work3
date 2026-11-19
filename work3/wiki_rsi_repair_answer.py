"""Test whether a frozen solver uses the verified return on W4b repair states."""

import html
import json
import os
import re
import time
import unicodedata
from collections import Counter

from work3 import wiki_rsi_repair_audit as audit
from work3 import wiki_rsi_repair_pool as pool
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import contains, sha256, text_key, title_key
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_decode import DecodeEngine
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w4c_repair_answer_v1"
AUDIT_CASES = "w4_repair_audit_cases.json"
AUDIT_SUMMARY = "wiki_rsi_w4_repair_audit_summary.json"
PREPARED = "w4_repair_answer_prepared.json"
PREP_SUMMARY = "wiki_rsi_w4_repair_answer_prepare_summary.json"
ARMS = ("prefix_only", "original_question", "related_foil", "target_year_masked", "target")


def normalized_docs(docs):
    return [[unicodedata.normalize("NFKC", html.unescape(value)) for value in doc] for doc in docs]


def mask_year(docs, year):
    pattern = re.compile(rf"(?<![^\W_]){re.escape(year)}(?![^\W_])")
    return [[pattern.sub("[year omitted]", value) for value in doc] for doc in docs]


def render(tok, question, histories, extra):
    messages = [
        {"role": "system", "content":
         'Answer the question with a single four-digit year inside <answer> and </answer>. '
         'If you cannot determine the year, output <answer>unknown</answer>. Output only the tagged answer.'},
        {"role": "user", "content": f"Question: {question}\n\n"
         f"Initial search results (title, passage, in retrieval order):\n{json.dumps(histories, ensure_ascii=False)}"
         f"\n\nAdditional search results (title, passage, in retrieval order):\n{json.dumps(extra, ensure_ascii=False)}"},
    ]
    return w1.render_ids(tok, messages, tools=())


def load_cases(run_dir):
    dataset = w1.read_json(run_dir / AUDIT_CASES)
    summary = w1.read_json(run_dir / AUDIT_SUMMARY)
    frozen = w1.read_json(run_dir / pool.PREPARED)
    prep = w1.read_json(run_dir / pool.PREP_SUMMARY)
    if (dataset["protocol"] != audit.PROTOCOL or summary["protocol"] != audit.PROTOCOL
            or frozen["protocol"] != pool.PROTOCOL or prep["prepared_sha256"] != sha256(run_dir / pool.PREPARED)
            or summary["preparation_sha256"] != sha256(run_dir / pool.PREP_SUMMARY)
            or not summary["retriever"]["passed"] or summary["retriever"]["retriever_id"] != frozen["retriever_id"]):
        raise ValueError("Use the completed W4b audit and its W4a source preparation")
    selected = dataset["selected"]
    counts = Counter(case["split"] for case in selected)
    expected = Counter({split: summary["by_split"][split]["n"] for split in pool.SPLIT_CAPS})
    if not selected or len(selected) != summary["normalized_one_witness_page_disjoint"] or counts != expected:
        raise ValueError("W4b candidate counts or split totals changed")
    traces = {row["source"]: row for row in dataset["source_traces"]}
    if len(traces) != len(dataset["source_traces"]):
        raise ValueError("W4b contains duplicate source traces")
    used, seen = set(), []
    blocked = set(frozen["blocked"])
    for case in selected:
        source, target, foil, year = (case[key] for key in ("source", "target", "foil", "answer"))
        pages = {title_key(title) for title in (source, target, foil)}
        if len(pages) != 3 or pages & (blocked | used) or not re.fullmatch(r"[0-9]{4}", year):
            raise ValueError("W4b page isolation or year format changed")
        used.update(pages)
        prefix = traces[source]["prefix"]
        people, _ = pool.visible_credits(source, prefix, blocked)
        if (people.get(case["role"]) != target or foil not in people.values()
                or source not in frozen["sources"]
                or pool.has_year(prefix + [("", case["question"])], year)
                or contains(text_key(case["question"]), text_key(target))):
            raise ValueError("W4b state no longer supports the recorded visible role")
        positive = dataset["person_returns"][target]
        negative = dataset["person_returns"][foil]
        witness = audit.birth_entries(target, positive)["normalized"]
        if (not witness or witness[0] != year or witness[1] != case["target_witness"]
                or not any(title_key(doc[0]) == title_key(foil) for doc in negative)
                or any(title_key(doc[0]) == title_key(target) for doc in negative)
                or pool.has_year(negative, year)):
            raise ValueError("W4b positive or foil evidence no longer verifies")
        seen.append((case, prefix, positive, negative))
    return seen


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    cases = load_cases(run_dir)
    checkpoint = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, checkpoint)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    records = []
    lengths = {arm: [] for arm in ARMS}
    for index, (case, prefix, positive, negative) in enumerate(cases):
        year = case["answer"]
        history = normalized_docs(prefix)
        target = normalized_docs(positive)
        masked = mask_year(target, year)
        if masked == target or pool.has_year(masked, year):
            raise ValueError("Year-masked control did not remove the verified answer")
        docs = {"prefix_only": [], "original_question": normalized_docs(case["original_question"]["passages"]),
                "related_foil": normalized_docs(negative), "target_year_masked": masked, "target": target}
        prompts = {arm: render(tok, case["question"], history, docs[arm]) for arm in ARMS}
        if any(not ids or len(ids) + w1.MAX_NEW + 1 > w1.solver.MAX_MODEL_LEN for ids in prompts.values()):
            raise ValueError("A W4c prompt cannot retain the full ranked returns")
        for arm in ARMS:
            lengths[arm].append(len(prompts[arm]))
        records.append({"case": index, "split": case["split"], "role": case["role"], "year": year,
                        "prompts": prompts})
        progress.update("running", index + 1, len(cases), detail="preparing_repair_answers")
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "records": records})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "prepared_sha256": sha256(path),
               "w4b_cases_sha256": sha256(run_dir / AUDIT_CASES), "w4b_summary_sha256": sha256(run_dir / AUDIT_SUMMARY),
               "model_sha256": checkpoint["model_sha256"], "by_split": Counter(row["split"] for row in records),
               "prompt_tokens": {arm: {"min": min(sizes), "max": max(sizes), "mean": round(sum(sizes) / len(sizes), 1)}
                                 for arm, sizes in lengths.items()},
               "n": len(records), "planned_completions": len(records) * len(ARMS)}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(records), len(records), detail="ready", metrics={"completion_jobs": summary["planned_completions"]})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def score(output, year):
    match = re.fullmatch(r"<answer>([0-9]{4}|unknown)</answer>", output["text"].strip())
    valid = bool(match and output["finish"] == "stop")
    return {**output, "valid": valid, "correct": int(valid and match[1] == year),
            "abstain": int(valid and match[1] == "unknown")}


def summarize_group(rows):
    outcomes = [row["arms"] for row in rows]
    return {"n": len(rows), "arms": {arm: {metric: sum(row[arm][metric] for row in outcomes)
                                          for metric in ("valid", "correct", "abstain")} for arm in ARMS},
            "target_pairs": {arm: {
                "strict_em": w1.paired(outcomes, "target", arm),
                "both_valid": w1.paired(
                    [row for row in outcomes if row["target"]["valid"] and row[arm]["valid"]], "target", arm),
            } for arm in ARMS if arm != "target"}}


def run(args, run_dir, progress):
    path = run_dir / PREPARED
    data, ready = w1.read_json(path), w1.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or ready["protocol"] != PROTOCOL or ready["prepared_sha256"] != sha256(path)
            or ready["w4b_cases_sha256"] != sha256(run_dir / AUDIT_CASES)
            or ready["w4b_summary_sha256"] != sha256(run_dir / AUDIT_SUMMARY)):
        raise ValueError("Use the matching W4c preparation and completed W4b run")
    if not re.fullmatch(r"[0-9]+", os.environ.get("CUDA_VISIBLE_DEVICES", "")):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one idle physical GPU")
    w1.verify_checkpoint(args, ready)
    profile = w1.runtime_config(args)
    cases = data["records"]
    if (not cases or ready["n"] != len(cases) or ready["planned_completions"] != len(cases) * len(ARMS)
            or any(set(case["prompts"]) != set(ARMS) for case in cases)):
        raise ValueError("W4c planned cases or arms changed")
    start = time.monotonic()
    progress.update("running", 0, len(cases), detail="loading_solver")
    engine = DecodeEngine(args.model_dir, profile)
    loaded = time.monotonic()
    records, tokens = [], 0
    for offset in range(0, len(cases), 4):
        batch = cases[offset:offset + 4]
        prompts = [case["prompts"][arm] for case in batch for arm in ARMS]
        responses = engine.generate(prompts)
        if len(responses) != len(prompts):
            raise RuntimeError("W4c decoder did not return every paired completion")
        outputs = iter(responses)
        for case in batch:
            arms = {arm: score(next(outputs), case["year"]) for arm in ARMS}
            tokens += sum(output["tokens"] for output in arms.values())
            records.append({key: case[key] for key in ("case", "split", "role")} | {"arms": arms})
        progress.update("running", len(records), len(cases), detail="generating",
                        metrics={"generated_tokens": tokens, "completion_jobs": len(records) * len(ARMS)})
    atomic_json(run_dir / "w4_repair_answer_cases.json", {"protocol": PROTOCOL, "records": records})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "runtime": profile,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "completion_jobs": len(records) * len(ARMS),
               "generated_tokens": tokens, "load_seconds": round(loaded - start, 3),
               "wall_seconds": round(time.monotonic() - start, 3), "all_cases": summarize_group(records),
               "by_split": {name: summarize_group([row for row in records if row["split"] == name])
                            for name in pool.SPLIT_CAPS}}
    atomic_json(run_dir / "wiki_rsi_w4_repair_answer_summary.json", summary)
    progress.update("done", len(records), len(cases), detail="completed",
                    metrics={"completion_jobs": summary["completion_jobs"], "generated_tokens": tokens})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
