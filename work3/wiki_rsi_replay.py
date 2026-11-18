"""Prepare and run the frozen same-history Wikipedia evidence replay."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from bisect import bisect_left
from collections import Counter, defaultdict, deque
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from work3 import p1_agent as solver
from work3 import wiki_rsi_audit as audit
from work3.evidence_ref import answer_keys, contains, question_key, sha256, text_key, title_key
from work3.p0_retrieval_diag import (
    DOC, EMPTY, ROOT, SUPPORT_DATASETS, TOOL, TraceError, dataset_name, question_of, resolve_dump,
    revision, test_questions, trace_of,
)
from work3.wiki_rsi_progress import Progress, atomic_json, progress_root
from work3.wiki_rsi_references import WIKI_RSI_REFERENCES

PIN = "98c4a6dd2495e85d8b4d947ab4c030e540c922c5"
PROTOCOL = "wiki_rsi_w1_v1"
MAX_NEW = 192
MIN_PER_DATASET = 40
NEGATIVE_POOL = 1500
ANY_ANSWER_TAG = re.compile(r"</?answer>", re.IGNORECASE)
ANY_TOOL_TAG = re.compile(r"</?tool_(?:call|response)>", re.IGNORECASE)
ROLE = re.compile(r"^\nassistant\n")
RISKY_TEXT = re.compile(r"<\||</?(?:tool_call|tool_response|answer|think)>")
ARMS = ("original_replay", "empty", "irrelevant", "evidence")
INSTRUCTION = (
    "Use the question and the existing search history. The supplementary Wikipedia passages "
    "below may be empty or irrelevant. Answer now without another search. "
    "Put only your short final answer inside <answer> and </answer>.\n\n"
    "Supplementary Wikipedia passages:\n"
)


class Excluded(ValueError):
    """A case fails a preregistered preparation check."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    with path.open(encoding="utf-8") as stream:
        yield from ((number, json.loads(line)) for number, line in enumerate(stream, 1)
                    if line.strip())


def input_files(args):
    drzero = args.drzero.resolve()
    test = args.test_parquet or drzero / "data/test.parquet"
    if revision(drzero) != PIN:
        raise ValueError("DRZERO_DIR must point to the pinned Dr. Zero checkout")
    if not test.is_file():
        raise FileNotFoundError(f"Original P0 test parquet not found: {test}")
    run_root = Path(os.environ.get("OUT_ROOT") or drzero / "work3_p0")
    dump = resolve_dump(args.eval_dump or run_root / "eval/iter3")
    return dump, test


def model_fingerprint(model):
    files = sorted(set(model.glob("*.safetensors")) | set(model.glob("pytorch_model*.bin")))
    if not files or not (model / "config.json").is_file():
        raise ValueError("Use the converted local HF directory of the original P0 iteration-3 solver")
    files += [model / name for name in (
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "special_tokens_map.json", "model.safetensors.index.json",
        "pytorch_model.bin.index.json", "added_tokens.json", "vocab.json", "merges.txt",
    ) if (model / name).is_file()]
    return {path.name: sha256(path) for path in files}


def render_ids(tok, messages, *, gen=True, tools=(solver.TOOL_SCHEMA,)):
    prompt = tok.apply_chat_template(
        messages, tools=list(tools) or None, add_generation_prompt=gen, tokenize=False)
    return list(tok(prompt)["input_ids"])


def reconstruct(tok, initial, row):
    expected_input = tok.decode(render_ids(tok, initial, tools=()), skip_special_tokens=True)
    if expected_input != row["input"]:
        raise Excluded("original_input_mismatch")
    output = row["output"]
    try:
        trace = trace_of(output)
    except TraceError as error:
        raise Excluded("invalid_tool_trace") from error
    if trace.errors or not trace.queries:
        raise Excluded("invalid_tool_trace")

    prefix = list(initial)
    cursor = 0
    for turn, (name, events) in enumerate(itertools.groupby(TOOL.finditer(output), key=lambda e: e[1])):
        group = list(events)
        lead = output[cursor:group[0].start()]
        gaps = [output[left.end():right.start()] for left, right in zip(group, group[1:])]
        if any(gap.strip() for gap in gaps):
            raise Excluded("ambiguous_tool_group")
        if turn % 2 == 0 and name == "tool_call":
            if turn:
                match = ROLE.match(lead)
                if not match:
                    raise Excluded("assistant_boundary_missing")
                lead = lead[match.end():]
            calls = [{"id": str(index), "type": "function", "function": json.loads(event[2])}
                     for index, event in enumerate(group)]
            if len(calls) > 64 or any(
                set(call["function"]) != {"name", "arguments"} or
                set(call["function"]["arguments"]) != {"query_list"} for call in calls
            ):
                raise Excluded("unsupported_call_schema")
            prefix.append(solver.message("assistant", lead.removesuffix("\n"), calls))
        elif turn % 2 == 1 and name == "tool_response" and lead == "\nuser\n":
            if len(group) != len(prefix[-1]["tool_calls"]):
                raise Excluded("unpaired_tool_messages")
            prefix.extend(solver.message("tool", event[2].removeprefix("\n").removesuffix("\n"))
                          for event in group)
        else:
            raise Excluded("ambiguous_tool_group")
        cursor = group[-1].end()

    ending = output[cursor:]
    match = ROLE.match(ending)
    if prefix[-1]["role"] != "tool" or not match:
        raise Excluded("final_assistant_boundary_missing")
    final = ending[match.end():].removesuffix("\n")
    if not final.strip() or re.search(r"(?m)^(system|user|assistant)$", final):
        raise Excluded("ambiguous_final_assistant_turn")
    if ANY_ANSWER_TAG.search(output[:cursor]):
        raise Excluded("answer_tag_in_retained_history")
    if sum(message["role"] == "assistant" for message in prefix) >= solver.MAX_TURNS:
        raise Excluded("turn_budget")

    initial_ids = render_ids(tok, initial)
    boundary = render_ids(tok, prefix)
    whole = render_ids(tok, prefix + [solver.message("assistant", final)], gen=False)
    if (whole[:len(initial_ids)] != initial_ids or whole[:len(boundary)] != boundary or
            tok.decode(whole[len(initial_ids):], skip_special_tokens=True) != output):
        raise Excluded("transcript_round_trip_mismatch")
    return prefix, boundary, final, trace


def original_answer(output):
    answers = re.findall(solver.ANSWER_PATTERN, output, re.DOTALL)
    return answers[-1].strip() if answers else None


def initial_messages(test, wanted):
    from pyarrow.parquet import ParquetFile

    prompts = {}
    seen = set()
    for batch in ParquetFile(test).iter_batches(columns=["data_source", "prompt"]):
        for row in batch.to_pylist():
            ds = dataset_name(row["data_source"])
            if ds not in SUPPORT_DATASETS:
                continue
            prompt = row["prompt"]
            if not prompt or prompt[-1]["role"] != "user":
                continue
            key = ds, question_key(question_of(prompt[-1]["content"]))
            if key in wanted:
                if key in seen:
                    prompts.pop(key, None)
                else:
                    prompts[key] = prompt
                seen.add(key)
    return prompts


def raw_passages(output):
    query_number = 0
    pending = deque()
    for event in TOOL.finditer(output):
        if event[1] == "tool_call":
            pending.append(len(json.loads(event[2])["arguments"]["query_list"]))
            continue
        query_count = pending.popleft()
        result = json.loads(event[2])["result"]
        if result == EMPTY:
            query_number += query_count
            continue
        for block in re.split(r"\n[ \t]*---[ \t]*\n", result.replace("\r\n", "\n")):
            query_number += 1
            headers = list(DOC.finditer(block))
            for index, header in enumerate(headers):
                end = headers[index + 1].start() if index + 1 < len(headers) else len(block)
                body = block[header.end():end].strip()
                if body:
                    yield (query_number, index + 1), (header[2], body)


def catalog(dump, cases):
    target_lines = {case["dump_line"] for case in cases}
    sources = defaultdict(set)
    for case in cases:
        for line, query, rank in case["support_passages"]:
            sources[line].add((query, rank))
    negative = {ds: {} for ds in SUPPORT_DATASETS}
    rows, passages = {}, {}
    for line, item in read_lines(dump):
        ds = dataset_name(item["data_source"])
        if line in target_lines:
            rows[line] = item
        take = ds in negative and line not in target_lines and len(negative[ds]) < NEGATIVE_POOL
        if line not in sources and not take:
            continue
        try:
            trace = trace_of(item["output"])
        except TraceError:
            continue
        if trace.errors:
            continue
        for coord, passage in raw_passages(item["output"]):
            if coord in sources.get(line, ()):
                passages[(line, *coord)] = passage
            if take and len(negative[ds]) < NEGATIVE_POOL and not any(
                RISKY_TEXT.search(text) for text in passage
            ):
                normalized = title_key(passage[0]), text_key(passage[1])
                negative[ds].setdefault(normalized, passage)
    return rows, passages, negative


def doc_lengths(tok, passages):
    formatted = [solver.passages_to_string(["\n".join(doc)]) for doc in passages]
    return [len(ids) for ids in tok(formatted, add_special_tokens=False)["input_ids"]]


def choose_negative(tok, pool, evidence, banned_titles, aliases):
    normalized_aliases = {text_key(alias) for alias in aliases}
    eligible = [(length, title, passage) for length, title, body, passage in pool
                if title not in banned_titles and not any(
                    contains(body, alias) or contains(text_key(passage[0]), alias)
                    for alias in normalized_aliases)]
    lengths = [item[0] for item in eligible]
    chosen = []
    for size in doc_lengths(tok, evidence):
        pos = bisect_left(lengths, size)
        options = [index for index in (pos - 1, pos) if 0 <= index < len(eligible)]
        if not options:
            raise Excluded("no_length_matched_control")
        best = min(options, key=lambda index: abs(lengths[index] - size))
        if abs(lengths[best] - size) > max(4, 0.1 * size):
            raise Excluded("no_length_matched_control")
        chosen.append(eligible[best][2])
        title = eligible[best][1]
        eligible = [item for item in eligible if item[1] != title]
        lengths = [item[0] for item in eligible]
    return chosen


def with_passages(tok, prefix, docs):
    content = INSTRUCTION + json.dumps(
        [solver.passages_to_string(["\n".join(doc)]) for doc in docs], ensure_ascii=False)
    return render_ids(tok, prefix + [solver.message("user", content)])


def prepare_case(tok, item, initial, case, aliases, reference, sources, pool):
    if (not initial or not aliases or dataset_name(item["data_source"]) != case["dataset"] or
            question_key(question_of(item["input"])) != case["question_key"] or
            float(item["score"]) != 0 or not reference or
            not (set(reference["answers"]) & answer_keys(aliases))):
        raise Excluded("manifest_case_mismatch")
    prefix, boundary, final, trace = reconstruct(tok, initial, item)
    recorded = original_answer(final)
    if recorded is not None and solver.em_check(recorded, aliases) != 0:
        raise Excluded("original_score_mismatch")
    visible = defaultdict(list)
    for group in trace.queries:
        for title, body in group:
            visible[title].append(body)
    missing = {(title, sentence) for title, sentence in audit.support_facts(reference)
               if not any(contains(body, sentence) for body in visible.get(title, []))}
    expected = case["cohort"] == "missing_with_source"
    if len(missing) != case["missing_sentences"] or expected != bool(missing):
        raise Excluded("support_gap_mismatch")

    prompts = {"original_replay": boundary, "empty": with_passages(tok, prefix, [])}
    if expected:
        evidence = [sources.get(tuple(locator)) for locator in case["support_passages"]]
        if not evidence or any(
            doc is None or any(RISKY_TEXT.search(text) for text in doc) for doc in evidence
        ) or not all(any(title_key(doc[0]) == title and contains(text_key(doc[1]), sentence)
                         for doc in evidence) for title, sentence in missing):
            raise Excluded("source_passage_mismatch")
        banned = set(reference["titles"]) | set(visible)
        irrelevant = choose_negative(tok, pool, evidence, banned, aliases)
        prompts["evidence"] = with_passages(tok, prefix, evidence)
        prompts["irrelevant"] = with_passages(tok, prefix, irrelevant)
        added = len(prompts["evidence"]) - len(prompts["empty"])
        if abs(len(prompts["evidence"]) - len(prompts["irrelevant"])) > max(8, 0.1 * added):
            raise Excluded("no_length_matched_control")
    if max(map(len, prompts.values())) + MAX_NEW + 1 > solver.MAX_MODEL_LEN:
        raise Excluded("context_budget")
    return {"dataset": case["dataset"], "cohort": case["cohort"], "qid": case["question_key"],
            "aliases": aliases, "original_answer": recorded, "prompts": prompts}


def prepare(args, run_dir, progress):
    dump, test = input_files(args)
    audit_dir = run_dir / "w0"
    baseline = read_json(audit_dir / "wiki_rsi_input_summary.json")
    manifest = audit_dir / "wiki_rsi_manifest.jsonl"
    file_hashes = {"dump_sha256": sha256(dump), "test_sha256": sha256(test),
                   "references_sha256": sha256(args.references)}
    if any(baseline["provenance"].get(key) != digest for key, digest in file_hashes.items()):
        raise ValueError("W0 input hashes differ; run a new audit with the intended P0 inputs")
    cases = [case for _, case in read_lines(manifest)]
    identities = {(case["dataset"], case["question_key"]) for case in cases}
    if len(identities) != len(cases):
        raise ValueError("Duplicate question in W0 manifest")
    for ds in SUPPORT_DATASETS:
        for cohort in audit.CAPS:
            if sum(case["dataset"] == ds and case["cohort"] == cohort for case in cases) != (
                baseline["by_dataset"][ds].get(f"selected_{cohort}", 0)
            ):
                raise ValueError("W0 manifest and summary counts differ")
    progress.update("running", total=len(cases), detail="hashing_checkpoint")
    checkpoint = model_fingerprint(args.model_dir)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tests, _ = test_questions(test)
    initial = initial_messages(test, identities)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = json.load(stream)["datasets"]
    rows, sources, negatives = catalog(dump, cases)
    pools = {}
    for ds, entries in negatives.items():
        lengths = doc_lengths(tok, list(entries.values())) if entries else []
        pools[ds] = sorted((length, title, body, doc) for length, ((title, body), doc)
                           in zip(lengths, entries.items()))

    prepared = []
    counts = {ds: Counter() for ds in SUPPORT_DATASETS}
    for number, case in enumerate(cases, 1):
        ds, qid = case["dataset"], case["question_key"]
        item = rows.get(case["dump_line"])
        try:
            if item is None:
                raise Excluded("missing_dump_line")
            record = prepare_case(tok, item, initial.get((ds, qid)), case, tests[ds].get(qid),
                                  refs[ds]["items"].get(qid), sources, pools[ds])
        except Excluded as error:
            counts[ds][f"{case['cohort']}:excluded:{error}"] += 1
        else:
            prepared.append(record)
            counts[ds][case["cohort"]] += 1
        if number % 5 == 0 or number == len(cases):
            progress.update("running", number, len(cases), detail="checking_history")
    report = {
        "protocol": PROTOCOL, "repo_revision": revision(ROOT), "drzero_revision": PIN,
        "transformers_version": version("transformers"), "model_sha256": checkpoint,
        "source": {**file_hashes, "manifest_sha256": sha256(manifest)},
        "w0_by_dataset": baseline["by_dataset"], "by_dataset": counts, "prepared_cases": len(prepared),
        "ready_for_full_replay": baseline["ready_for_replay_preparation"] and all(
            counts[ds]["missing_with_source"] >= MIN_PER_DATASET for ds in SUPPORT_DATASETS),
    }
    atomic_json(run_dir / "prepared.json", {"protocol": PROTOCOL, "records": prepared})
    report["prepared_sha256"] = sha256(run_dir / "prepared.json")
    atomic_json(run_dir / "wiki_rsi_prepare_summary.json", report)
    progress.update("done" if report["ready_for_full_replay"] else "blocked", len(cases), len(cases),
                    detail="ready" if report["ready_for_full_replay"] else "insufficient_pairs",
                    metrics={ds: counts[ds]["missing_with_source"] for ds in SUPPORT_DATASETS})
    print(json.dumps(report, indent=2), flush=True)
    return 0


def score_generation(text, aliases, finish):
    pred = original_answer(text)
    valid = (bool(pred) and text.count("<answer>") == 1 and text.count("</answer>") == 1
             and not ANY_TOOL_TAG.search(text)
             and text.rstrip().endswith("</answer>") and finish == "stop")
    return {"answer": pred, "valid": bool(valid),
            "correct": int(bool(valid) and solver.em_check(pred, aliases)),
            "p0_em": int(pred is not None and solver.em_check(pred, aliases))}


def paired(rows, treatment, control):
    differences = np.array([row[treatment]["correct"] - row[control]["correct"] for row in rows])
    n = len(differences)
    if not n:
        return {"n": 0, "gain": None, "ci95": None, "wins": 0, "losses": 0, "p_exact": None}
    wins = int((differences == 1).sum())
    losses = int((differences == -1).sum())
    discordant = wins + losses
    boot = np.random.default_rng(audit.SEED).multinomial(n, [losses / n, wins / n,
                                                          (n - discordant) / n], size=10000)
    interval = np.quantile((boot[:, 1] - boot[:, 0]) / n, [.025, .975]).tolist()
    p_value = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
                  / (2 ** discordant))
    return {"n": n, "gain": float(differences.mean()), "ci95": interval,
            "wins": wins, "losses": losses, "p_exact": p_value}


def summarize(records):
    by_dataset = {}
    for ds in SUPPORT_DATASETS:
        dataset = [row for row in records if row["dataset"] == ds]
        missing = [row["arms"] for row in dataset if row["cohort"] == "missing_with_source"]
        comparable = [row for row in missing if all(row[arm]["valid"] for arm in ARMS[1:])]
        grouped = {cohort: {} for cohort in audit.CAPS}
        for cohort in grouped:
            for arm in ARMS:
                results = [row["arms"][arm] for row in dataset if
                           row["cohort"] == cohort and arm in row["arms"]]
                if results:
                    grouped[cohort][arm] = {
                        "n": len(results), "valid": sum(result["valid"] for result in results),
                        "correct": sum(result["correct"] for result in results),
                        "finish": dict(Counter(result["finish"] for result in results)),
                    }
        by_dataset[ds] = {
            "missing_pair_cases": len(missing), "valid_triplets": len(comparable),
            "evidence_specific_rescues": sum(
                row["evidence"]["correct"] and not (row["empty"]["correct"] or
                                                    row["irrelevant"]["correct"])
                for row in comparable),
            "evidence_vs_irrelevant": paired(missing, "evidence", "irrelevant"),
            "evidence_vs_empty": paired(missing, "evidence", "empty"),
            "valid_triplets_evidence_vs_irrelevant": paired(comparable, "evidence", "irrelevant"),
            "valid_triplets_evidence_vs_empty": paired(comparable, "evidence", "empty"),
            "original_p0_wrong_cases": len(dataset),
            "original_replay_p0_em_correct": sum(row["arms"]["original_replay"]["p0_em"] for row in dataset),
            "original_replay_same_normalized_answer": sum(row["same_original_answer"] for row in dataset),
            "by_cohort": grouped,
        }
    return by_dataset


def replay(args, run_dir, progress):
    report = read_json(run_dir / "wiki_rsi_prepare_summary.json")
    path = run_dir / "prepared.json"
    if report["protocol"] != PROTOCOL or report["prepared_sha256"] != sha256(path):
        raise ValueError("Prepared cases failed provenance check")
    if args.phase == "replay" and not report["ready_for_full_replay"]:
        progress.update("blocked", detail="insufficient_pairs")
        print("Full replay blocked by the W0/W1 preparation gate; inspect wiki_rsi_prepare_summary.json.",
              file=sys.stderr, flush=True)
        return 2
    cases = read_json(path)["records"]
    if args.phase == "smoke":
        cases = [row for ds in SUPPORT_DATASETS for row in
                 [case for case in cases if case["dataset"] == ds and
                  case["cohort"] == "missing_with_source"][:2]]
    if not cases:
        raise ValueError("No prepared missing-evidence cases")
    progress.update("running", total=len(cases), detail="checking_checkpoint")
    if model_fingerprint(args.model_dir) != report["model_sha256"]:
        raise ValueError("Replay model fingerprint differs from preparation")
    from vllm import LLM, SamplingParams

    progress.update("running", total=len(cases), detail="loading_solver")
    start = time.monotonic()
    model = LLM(model=str(args.model_dir), tensor_parallel_size=args.tp,
                max_model_len=solver.MAX_MODEL_LEN, dtype="bfloat16",
                gpu_memory_utilization=args.gpu_memory, enforce_eager=True, generation_config="vllm")
    load_seconds = time.monotonic() - start
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW,
                              stop_token_ids=[], skip_special_tokens=True)
    records = []
    start = time.monotonic()
    generated_tokens = prompt_tokens = 0
    batch_size = 8
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset + batch_size]
        jobs = [(case, arm) for case in batch for arm in ARMS if arm in case["prompts"]]
        prompts = [{"prompt_token_ids": case["prompts"][arm]} for case, arm in jobs]
        outputs = model.generate(prompts, sampling_params=sampling, use_tqdm=False)
        by_case = defaultdict(dict)
        for (case, arm), output in zip(jobs, outputs, strict=True):
            if list(output.prompt_token_ids) != case["prompts"][arm] or len(output.outputs) != 1:
                raise RuntimeError("vLLM returned a different tokenized prompt or completion count")
            generated = output.outputs[0]
            prompt_tokens += len(output.prompt_token_ids)
            generated_tokens += len(generated.token_ids)
            by_case[case["qid"], case["dataset"]][arm] = {
                **score_generation(generated.text, case["aliases"], generated.finish_reason),
                "text": generated.text, "finish": generated.finish_reason,
                "prompt_tokens": len(output.prompt_token_ids), "generated_tokens": len(generated.token_ids),
            }
        for case in batch:
            arms = by_case[case["qid"], case["dataset"]]
            replay_answer = arms["original_replay"]["answer"]
            original = case["original_answer"]
            same = (solver.normalize_answer(replay_answer) == solver.normalize_answer(original)
                    if replay_answer is not None and original is not None else replay_answer == original)
            records.append({"dataset": case["dataset"], "cohort": case["cohort"], "qid": case["qid"],
                            "same_original_answer": same, "arms": arms})
        atomic_json(run_dir / f"{args.phase}_cases.json", {"protocol": PROTOCOL, "records": records})
        progress.update("running", len(records), len(cases), detail="generating",
                        metrics={"generated_tokens": generated_tokens, "prompt_tokens": prompt_tokens})
    wall_seconds = time.monotonic() - start
    metrics = {"wall_seconds": round(wall_seconds, 2), "generated_tokens": generated_tokens,
               "prompt_tokens": prompt_tokens}
    result = {
        "protocol": PROTOCOL, "mode": args.phase, "repo_revision": revision(ROOT),
        "drzero_revision": PIN, "prepared_sha256": report["prepared_sha256"],
        "prepare_summary_sha256": sha256(run_dir / "wiki_rsi_prepare_summary.json"),
        "runtime": {"vllm": version("vllm"), "transformers": version("transformers"),
                    "tensor_parallel_size": args.tp, "gpu_memory_utilization": args.gpu_memory,
                    "checkpoint_load_seconds": round(load_seconds, 2), "temperature": 0,
                    "max_new_tokens": MAX_NEW, "max_model_len": solver.MAX_MODEL_LEN,
                    "generation_gpu_hours": round(wall_seconds * args.tp / 3600, 5), **metrics},
        "by_dataset": summarize(records),
    }
    atomic_json(run_dir / f"wiki_rsi_{args.phase}_summary.json", result)
    triplets = sum(ds["valid_triplets"] for ds in result["by_dataset"].values())
    progress.update("done", len(records), len(cases), detail="completed",
                    metrics={**metrics, "valid_triplets": triplets})
    print(json.dumps(result, indent=2), flush=True)
    return 0


def run_audit(args, run_dir, progress):
    dump, test = input_files(args)
    audit_dir = run_dir / "w0"
    with TemporaryDirectory(prefix=".audit-", dir=run_dir) as temporary:
        staging = Path(temporary) / "w0"
        audit.main(["--eval-dump", str(dump), "--test-parquet", str(test),
                    "--references", str(args.references), "--out-dir", str(staging)])
        staging.rename(audit_dir)
    summary = read_json(audit_dir / "wiki_rsi_input_summary.json")
    metrics = {ds: summary["by_dataset"][ds].get("selected_missing_with_source", 0)
               for ds in SUPPORT_DATASETS}
    ready = summary["ready_for_replay_preparation"]
    progress.update("done" if ready else "blocked", 1, 1,
                    detail="ready" if ready else "insufficient_candidates", metrics=metrics)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "prepare", "smoke", "replay"))
    parser.add_argument("--run", required=True, help="run directory name under WIKI_RSI_ROOT")
    parser.add_argument("--drzero", type=Path, default=os.environ.get("DRZERO_DIR"))
    parser.add_argument("--eval-dump", type=Path, default=os.environ.get("P0_EVAL_DUMP"))
    parser.add_argument("--test-parquet", type=Path, default=os.environ.get("P0_TEST_PARQUET"))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("P0_ITER3_MODEL"))
    parser.add_argument("--references", type=Path, default=WIKI_RSI_REFERENCES)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--gpu-memory", type=float, default=0.70)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run):
        parser.error("--run requires a simple directory name")
    if not 0 < args.gpu_memory < 1:
        parser.error("--gpu-memory must be between zero and one")
    run_dir = progress_root() / args.run
    if args.phase == "audit":
        run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    elif not run_dir.is_dir():
        parser.error("Run the audit phase first")
    import fcntl

    with (run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another stage is active for this run")
        progress = Progress(run_dir, args.phase)
        if args.phase == "audit" and (run_dir / "w0").exists():
            parser.error("Audit outputs already exist; use a new run name to audit again")
        if args.phase == "prepare" and (run_dir / "wiki_rsi_prepare_summary.json").exists():
            parser.error("Preparation outputs already exist; use the same run for smoke or replay")
        if (run_dir / f"wiki_rsi_{args.phase}_summary.json").exists():
            parser.error("Stage already completed; use a new run name for an independent repeat")
        progress.update("running", detail="checking_inputs")
        try:
            if args.phase in {"audit", "prepare"} and args.drzero is None:
                raise ValueError("Set DRZERO_DIR to the pinned local Dr. Zero checkout")
            if args.phase != "audit" and args.model_dir is None:
                raise ValueError("Set P0_ITER3_MODEL to the original P0 iteration-3 HF solver")
            handlers = {"audit": run_audit, "prepare": prepare, "smoke": replay, "replay": replay}
            return handlers[args.phase](args, run_dir, progress)
        except (Exception, KeyboardInterrupt) as error:
            phase = progress.data["phases"][args.phase]
            progress.update("failed", phase["done"], phase["total"], detail=type(error).__name__)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
