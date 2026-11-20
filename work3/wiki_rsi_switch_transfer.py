"""Train retrieval preferences without filtering on the reader; measure natural-query transfer."""

import json
import os
import re
import time
from pathlib import Path
from tempfile import mkdtemp

from work3 import wiki_rsi_action_overnight as parallel
from work3 import wiki_rsi_natural_query as query_format
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_switch_data as data
from work3 import wiki_rsi_two_query_live as live
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import sha256, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_evidence_switch_transfer_v1"
PREPARED = "switch_transfer_prepared.json"
PREP_SUMMARY = "wiki_rsi_switch_transfer_prepare_summary.json"
SUMMARY = "wiki_rsi_switch_transfer_summary.json"
BASELINE = ("frozen_follow_up", "linked_context", *live.POSITIVE)


def date_rule(row):
    """Select the entity whose displayed passage has no explicit year."""
    choices = []
    for state in row["states"]:
        seen = {title_key(title) for title, passage in state
                if re.search(r"\b(?:1[5-9]|20)\d{2}\b", passage)}
        missing = [i for i, title in enumerate(row["pages"]) if title_key(title) not in seen]
        choices.append(missing[0] if len(missing) == 1 else 0)
    return choices


def tallies(outcome):
    return {key: int(outcome[key]) for key in ("correct", "evidence_complete", "queries")}


def natural_examples(tok, cases, recorded):
    rows = []
    for case, result in zip(cases, recorded, strict=True):
        columns = ("dataset", "qid", "title_stratum")
        if any(case[key] != result[key] for key in columns):
            raise ValueError("The W7b natural case identity or ordering changed")
        arms, by_query = result["arms"], {}
        for arm in BASELINE:
            query = arms[arm]["second_query"]
            if query and query != result["first_query"]:
                by_query.setdefault(query, tallies(arms[arm]))
        prompts = [tok(query, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for query in by_query]
        context = [*case["visible"], *result["first_passages"]]
        ids, _ = live.pack(tok, case["question"], context, query_format.prompt_ids, query_format.MAX_NEW + 1)
        outcomes = [outcome for tokens, outcome in zip(prompts, by_query.values(), strict=True)
                    if 1 < len(tokens) <= query_format.MAX_NEW + 1]
        completions = [tokens for tokens in prompts if 1 < len(tokens) <= query_format.MAX_NEW + 1]
        rows.append({key: case[key] for key in columns} | {
            "prompt": ids, "completions": completions, "outcomes": outcomes,
            "first_only": tallies(arms["first_only"]), "frozen_follow_up": tallies(arms["frozen_follow_up"])})
    return rows


def verify_natural_totals(recorded, summary):
    for dataset in SUPPORT_DATASETS:
        group = [row["arms"] for row in recorded if row["dataset"] == dataset]
        expected = summary[dataset]["all"]
        if (len(group) != expected["n"] or any(
                sum(row[arm][key] for row in group) != expected["arms"][arm][key]
                for arm in live.ARMS for key in ("correct", "evidence_complete", "queries", "valid"))):
            raise ValueError("Cached natural outcomes disagree with the verified W7b summary")


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    from work3 import wiki_rsi_evidence_switch as switch
    from work3 import wiki_rsi_switch_train as train

    prior = replay.read_json(run_dir / train.SUMMARY)
    source, ready = (replay.read_json(run_dir / name) for name in (data.PREPARED, data.PREP_SUMMARY))
    w7 = replay.read_json(run_dir / live.SUMMARY)
    w7_inputs = replay.read_json(run_dir / live.PREP_SUMMARY)
    if (any(row["protocol"] != data.PROTOCOL for row in (prior, source, ready))
            or prior["decision"] != "insufficient_reader_verified_pairs"
            or prior["preparation_sha256"] != sha256(run_dir / data.PREP_SUMMARY)
            or ready["prepared_sha256"] != sha256(run_dir / data.PREPARED)
            or ready["w7b_sha256"] != sha256(run_dir / live.SUMMARY)
            or w7["preparation_sha256"] != sha256(run_dir / live.PREP_SUMMARY)
            or w7_inputs["prepared_sha256"] != sha256(run_dir / live.PREPARED)):
        raise ValueError("Use the original W-switch and W7b artifacts")
    folder = run_dir / prior["worker_directory"]
    if folder.parent != run_dir or not folder.name.startswith("switch_workers_") or folder.is_symlink():
        raise ValueError("Invalid original reader worker directory")
    encoded_path = folder / train.ENCODED
    if prior["reader_gate"]["encoded_sha256"] != sha256(encoded_path):
        raise ValueError("The original held-out reader outcomes changed")
    original = replay.read_json(encoded_path)["records"]
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    replay.verify_checkpoint(args, ready)
    if tok.eos_token_id is None:
        raise ValueError("Expected the original tokenizer with an EOS token")
    groups, examples = {}, {}
    for split, rows in source["records"].items():
        examples[split], groups[split] = [], []
        for row in rows:
            if row["gains"] != [[0, 1], [1, 0]]:
                raise ValueError("Prepared source lacks the real retrieval preference reversal")
            encoded = switch.encode_example(tok, row)
            if encoded is not None:
                groups[split].append({**encoded, "relation": row["relation"]})
                examples[split].append(row)
    for current, saved in zip(groups["test"], original["test"], strict=True):
        if any(saved[key] != current[key] for key in ("prompts", "completions", "relation")):
            raise ValueError("The original held-out states no longer reproduce")
        current["reader"] = saved["reader"]
    natural = replay.read_json(run_dir / live.PREPARED)["cases"]
    recorded_path = run_dir / "w7_live_cases.json"
    recorded = replay.read_json(recorded_path)
    if recorded["protocol"] != live.PROTOCOL:
        raise ValueError("Use the original W7b responses")
    train_pages = {title_key(title) for row in examples["train"] for title in row["pages"]}
    test_pages = {title_key(title) for row in examples["test"] for title in row["pages"]}
    natural_pages = {title_key(title) for row in natural for title in row["support_titles"]}
    if train_pages & (test_pages | natural_pages):
        raise ValueError("Evaluated support pages intersect with training")
    verify_natural_totals(recorded["records"], w7["by_dataset_and_stratum"])
    groups["natural"] = natural_examples(tok, natural, recorded["records"])
    rule = [train.measure_choices(date_rule(row), encoded["reader"])
            for row, encoded in zip(examples["test"], groups["test"], strict=True)]
    if len(groups["train"]) < data.MIN_TRAIN or len(groups["test"]) < data.MIN_TEST:
        raise ValueError("Too few encodable retrieval-verified examples")
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "records": groups})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT), "prepared_sha256": sha256(path),
               "switch_sha256": sha256(run_dir / train.SUMMARY), "w7b_cases_sha256": sha256(recorded_path),
               "model_sha256": ready["model_sha256"], "rows": {split: len(rows) for split, rows in groups.items()},
               "simple_year_rule": {key: sum(row[key] for row in rule)
                                    for key in ("both_correct", "covered", "em")},
               "natural_pool_empty": sum(not row["completions"] for row in groups["natural"]),
               "new_physical_queries": 0}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(groups["natural"]), len(groups["natural"]), detail="ready")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def natural_policy(model, examples, pad, device):
    import torch

    from work3 import wiki_rsi_action_model as trainer

    model.eval()
    with torch.no_grad():
        return [row["outcomes"][int(trainer.mean_logprobs(model, row["prompt"], row["completions"],
                                                         pad, device).argmax())]
                if row["completions"] else row["first_only"] for row in examples]


def transfer_report(examples, receipts):
    policies = {"first_only": [row["first_only"] for row in examples],
                "frozen_follow_up": [row["frozen_follow_up"] for row in examples], **receipts}
    rows = [dict(zip(policies, values, strict=True)) for values in zip(*policies.values(), strict=True)]
    report = {}
    for dataset in SUPPORT_DATASETS:
        indices = [i for i, case in enumerate(examples) if case["dataset"] == dataset]
        records = [rows[index] for index in indices]
        report[dataset] = {"n": len(records),
            "arms": {arm: {key: sum(row[arm][key] for row in records)
                           for key in ("correct", "evidence_complete", "queries")} for arm in policies},
            "oracle_answer_count": sum(max((o["correct"] for o in examples[index]["outcomes"]),
                                          default=examples[index]["first_only"]["correct"]) for index in indices),
            "interaction_vs_ordinary": {
                seed: {field: replay.paired([{key: {"correct": row[key][field]} for key in
                                            (f"{seed}-interaction", f"{seed}-ordinary")} for row in records],
                                           f"{seed}-interaction", f"{seed}-ordinary")
                       for field in ("correct", "evidence_complete")}
                for seed in parallel.SEEDS}}
    return report


def run(args, run_dir, progress):
    from work3 import wiki_rsi_switch_train as train

    ready = replay.read_json(run_dir / PREP_SUMMARY)
    path = run_dir / PREPARED
    groups = replay.read_json(path)
    if (groups["protocol"] != PROTOCOL or ready["protocol"] != PROTOCOL
            or ready["prepared_sha256"] != sha256(path)
            or ready["switch_sha256"] != sha256(run_dir / train.SUMMARY)
            or ready["w7b_cases_sha256"] != sha256(run_dir / "w7_live_cases.json")):
        raise ValueError("Use the frozen transfer records")
    replay.verify_checkpoint(args, ready)
    devices = parallel.device_ids(os.environ.get("WIKI_RSI_SWITCH_GPUS", "0,1,2,3"))
    folder = Path(mkdtemp(prefix="switch_transfer_workers_", dir=run_dir))
    # Workers score cached returns, with no live index or reader calls.
    config = {"model_dir": str(args.model_dir), "encoded_path": str(path), "encoded_sha256": ready["prepared_sha256"]}
    start = time.monotonic()
    jobs = train.run_jobs(folder, config, devices, start + parallel.HOURS * 3600, progress)
    complete, results = train.job_results(folder, jobs, config["encoded_sha256"])
    report = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT),
              "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "worker_directory": folder.name,
              "jobs": jobs, "synthetic_test": results, "new_physical_queries": 0,
              "wall_seconds": round(time.monotonic() - start, 3),
              "decision": "natural_replay_complete" if complete else "incomplete_workers"}
    if complete:
        baselines = {"frozen_ranker": replay.read_json(folder / "frozen_natural.json")}
        selections = {name: row.pop("natural") for name, row in results.items()}
        report["frozen_test"] = replay.read_json(folder / "frozen.json")
        report["natural_replay"] = transfer_report(groups["records"]["natural"], baselines | selections)
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done" if complete else "blocked", sum(job["exit_code"] == 0 for job in jobs), len(jobs),
                    detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if complete else 2
