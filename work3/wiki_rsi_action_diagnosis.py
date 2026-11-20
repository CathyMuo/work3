"""Diagnose W6b query generation from cached Wikipedia results, with no model or retriever."""

import json
from collections import Counter

from work3 import wiki_rsi_action_overnight as overnight
from work3 import wiki_rsi_action_pilot as w6a
from work3 import wiki_rsi_action_train as w6b
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import sha256, title_key
from work3.p0_retrieval_diag import revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w6_query_diagnosis_v1"
SUMMARY = "wiki_rsi_w6_query_diagnosis_summary.json"
CASES = "w6_query_diagnosis_cases.json"
POLICIES = ("frozen", *(f"{seed}-{arm}" for seed, arm in overnight.JOBS))


def diagnose_action(case, action):
    query, docs = action["query"], action["passages"]
    if (action["status"] == "executed") != bool(query) or (not query and docs):
        raise ValueError("A recorded query and its execution status disagree")
    missing = case["missing_facts"]
    correct = w2.measure(docs, missing)["all_missing_facts"]
    if correct != action["correct"]:
        raise ValueError("Cached passages do not reproduce the recorded hit")
    related = {title_key(name) for name, _ in docs} & {title for title, _ in missing}
    outcome = ("hit" if correct else "missing_sentence" if related
               else "missing_page" if query else "not_executed")
    return {"query": query, "outcome": outcome,
            "literal_query_in_input": bool(query and w6a.literal_title(case["question"], case["visible"], query)),
            "exact_title_query": bool(query and title_key(query) == title_key(case["probe_title"]))}


def summarize(rows):
    def hit(row, policy):
        return row["policies"][policy]["outcome"] == "hit"

    hits = {policy: sum(hit(row, policy) for row in rows) for policy in POLICIES}
    covered = [row for row in rows if any(hit(row, policy) for policy in POLICIES)]
    missed = [row for row in rows if not any(hit(row, policy) for policy in POLICIES)]
    return {
        "n": len(rows), "single_policy_hits": hits,
        "known_title_hits": sum(row["known_title"] for row in rows),
        "any_policy_hits": len(covered),
        "any_policy_minus_best_single": len(covered) - max(hits.values()),
        "trained_only_hits": sum(not hit(row, "frozen") for row in covered),
        "frozen_only_hits": sum(hit(row, "frozen") and not any(hit(row, policy) for policy in POLICIES[1:])
                                for row in covered),
        "known_title_hit_all_policies_miss": sum(row["known_title"] for row in missed),
        "known_title_hit_no_policy_issued_title": sum(
            row["known_title"] and not any(action["exact_title_query"] for action in row["policies"].values())
            for row in missed),
        "by_policy": {policy: {
            "outcomes": dict(Counter(row["policies"][policy]["outcome"] for row in rows)),
            "literal_query_in_input": sum(row["policies"][policy]["literal_query_in_input"] for row in rows),
            "exact_title_query": sum(row["policies"][policy]["exact_title_query"] for row in rows),
            "lost_frozen_hits": sum(hit(row, "frozen") and not hit(row, policy) for row in rows),
            "new_hits_vs_frozen": sum(hit(row, policy) and not hit(row, "frozen") for row in rows),
        } for policy in POLICIES},
    }


def cached_inputs(run_dir):
    prepare_path, result_path = run_dir / overnight.PREP_SUMMARY, run_dir / overnight.SUMMARY
    ready, result = w1.read_json(prepare_path), w1.read_json(result_path)
    prepared_path = run_dir / overnight.PREPARED
    prepared = w1.read_json(prepared_path)
    if (any(obj["protocol"] != overnight.PROTOCOL for obj in (ready, result, prepared))
            or result["preparation_sha256"] != sha256(prepare_path)
            or ready["prepared_sha256"] != sha256(prepared_path)
            or ready["w6a_prepared_sha256"] != sha256(run_dir / w6a.PREPARED)
            or not result["retriever"]["passed"] or result["timed_out"]
            or set(result["matched_checkpoints"]) != {str(epoch) for epoch in overnight.CHECKPOINTS}
            or {(job["seed"], job["arm"]) for job in result["jobs"] if job["exit_code"] == 0} != set(overnight.JOBS)):
        raise ValueError("Use the complete W6b four-worker run and unmodified W6a inputs")
    worker_dir = result["worker_directory"]
    folder = run_dir / worker_dir
    if not worker_dir.startswith("w6_overnight_workers_") or folder.parent != run_dir or folder.is_symlink():
        raise ValueError("Invalid W6b worker directory")
    w6a_prepared, _ = w6b.prior_inputs(run_dir)
    cases = w6a_prepared["evaluation"]
    eval_baselines = w1.read_json(run_dir / w6b.EVAL_BASELINES)["records"]
    baseline_by_case = {(row["dataset"], row["qid"]): row for row in eval_baselines}
    for original, cached in zip(cases, prepared["evaluation"], strict=True):
        if any(original[k] != cached[k] for k in ("qid", "dataset", "title_stratum", "missing_facts")):
            raise ValueError("The W6b evaluation case order or content changed")
    return result, ready, prepared, folder, cases, baseline_by_case


def epoch_rows(folder, prepared, ready, epoch):
    checks = overnight.matched_report(folder, prepared["evaluation"], ready["prepared_sha256"], epoch)
    if checks is None:
        raise ValueError(f"The cached workers do not contain the complete epoch {epoch}")
    return {policy: w1.read_json(folder / (f"{policy}-{epoch}.json" if policy != "frozen" else "frozen.json"))["rows"]
            for policy in POLICIES}


def run(args, run_dir, progress):
    result, ready, prepared, folder, cases, baseline_by_case = cached_inputs(run_dir)
    outputs, reports = {}, {}
    for epoch in overnight.CHECKPOINTS:
        rows_by_policy = epoch_rows(folder, prepared, ready, epoch)
        rows = []
        for index, case in enumerate(cases):
            baseline = baseline_by_case[case["dataset"], case["qid"]]
            policies = {policy: diagnose_action(case, actions[index]) for policy, actions in rows_by_policy.items()}
            rows.append({key: case[key] for key in ("dataset", "qid", "title_stratum")} |
                        {"known_title": baseline["known_title"], "policies": policies})
        outputs[epoch] = rows
        grouped = {dataset: {stratum: summarize([row for row in rows if
                             (row["dataset"], row["title_stratum"]) == (dataset, stratum)])
                             for stratum in w6a.EVAL_CAPS} for dataset in w6b.SUPPORT_DATASETS}
        for seed, seed_report in result["matched_checkpoints"][str(epoch)]["by_seed"].items():
            for dataset, strata in seed_report["by_dataset_and_title_stratum"].items():
                for stratum, group in strata.items():
                    expected = {("frozen" if arm == "frozen" else f"{seed}-{arm}"): metrics["all_missing_facts"]
                                for arm, metrics in group["arms"].items()}
                    observed = grouped[dataset][stratum]["single_policy_hits"]
                    if group["n"] != grouped[dataset][stratum]["n"] or any(
                            observed[policy] != count for policy, count in expected.items()):
                        raise ValueError("Per-question records do not match the reported arm totals")
        reports[epoch] = grouped
        progress.update("running", len(reports), len(overnight.CHECKPOINTS), detail="diagnosing_cached_queries")
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT),
               "source_summary_sha256": sha256(run_dir / overnight.SUMMARY),
               "source_train_pairs": result["train_pairs"], "source_decision": result["decision"],
               "new_physical_queries": 0, "by_epoch": reports}
    atomic_json(run_dir / CASES, {"protocol": PROTOCOL, "by_epoch": outputs})
    atomic_json(run_dir / SUMMARY, summary)
    progress.update("done", len(reports), len(overnight.CHECKPOINTS), detail="completed")
    print(json.dumps(summary, indent=2), flush=True)
    return 0
