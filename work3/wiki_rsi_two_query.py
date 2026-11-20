"""Compare one learned or rule-based follow-up after the same cached frozen query."""

import json

from work3 import wiki_rsi_action_diagnosis as diagnosis
from work3 import wiki_rsi_action_overnight as overnight
from work3 import wiki_rsi_natural as natural
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import sha256
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w7_two_query_v1"
SUMMARY = "wiki_rsi_w7_two_query_summary.json"
EPOCH = overnight.CHECKPOINTS[-1]
MIN_GAIN = 5
RULES = tuple(f"rule_{arm}" for arm in natural.FIXED_ARMS)
TREATMENTS = diagnosis.POLICIES[1:]
POLICIES = ("frozen", *RULES, *TREATMENTS)


def replay(frozen, extra, missing):
    first, second = frozen["query"], extra["query"]
    duplicate = bool(second and second == first)
    take_extra = bool(second and not duplicate)
    docs = [*frozen["passages"], *(extra["passages"] if take_extra else [])]
    return {"hit": w2.measure(docs, missing)["all_missing_facts"],
            "queries": int(bool(first)) + int(take_extra), "duplicate": duplicate}


def summarize(rows):
    n = len(rows)
    frozen = sum(row["frozen"]["hit"] for row in rows)
    metrics = {policy: {
        "hits": sum(row[policy]["hit"] for row in rows),
        "queries": sum(row[policy]["queries"] for row in rows),
        "duplicates": sum(row[policy]["duplicate"] for row in rows),
        "lost_frozen_hits": sum(row["frozen"]["hit"] and not row[policy]["hit"] for row in rows),
    } for policy in POLICIES}
    best_rule = max(metrics[policy]["hits"] for policy in RULES)
    for row in metrics.values():
        row["net_vs_frozen"] = row["hits"] - frozen
        row["net_vs_best_rule"] = row["hits"] - best_rule
    return {"n": n, "rule_best_hits": best_rule, "policies": metrics}


def passed(report):
    return all(report[dataset]["all"]["policies"][f"{seed}-positive_only"]["net_vs_best_rule"] >= MIN_GAIN
               for dataset in SUPPORT_DATASETS for seed in overnight.SEEDS)


def run(args, run_dir, progress):
    check_path = run_dir / diagnosis.SUMMARY
    audit = w1.read_json(check_path)
    if (audit["protocol"] != diagnosis.PROTOCOL
            or audit["source_summary_sha256"] != sha256(run_dir / overnight.SUMMARY)
            or audit["new_physical_queries"] != 0):
        raise ValueError("Use the W6c diagnosis for the original cached W6b run")
    _, ready, prepared, folder, cases, baselines = diagnosis.cached_inputs(run_dir)
    by_policy = diagnosis.epoch_rows(folder, prepared, ready, EPOCH)
    by_group = {dataset: {key: [] for key in ("all", *diagnosis.w6a.EVAL_CAPS)} for dataset in SUPPORT_DATASETS}
    for index, case in enumerate(cases):
        dataset, qid = case["dataset"], case["qid"]
        missing = case["missing_facts"]
        recorded = {policy: actions[index] for policy, actions in by_policy.items()}
        for action in recorded.values():
            diagnosis.diagnose_action(case, action)
        frozen = recorded["frozen"]
        none = {"query": None, "passages": []}
        baseline = baselines[dataset, qid]["returned"]
        rules = {f"rule_{name}": {"query": case["queries"][name], "passages": baseline[case["queries"][name]]}
                 for name in natural.FIXED_ARMS}
        options = {"frozen": none, **rules, **{policy: recorded[policy] for policy in TREATMENTS}}
        row = {policy: replay(frozen, extra, missing) for policy, extra in options.items()}
        by_group[dataset]["all"].append(row)
        by_group[dataset][case["title_stratum"]].append(row)
    groups = {dataset: {name: summarize(rows) for name, rows in strata.items()} for dataset, strata in by_group.items()}
    report = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "w6c_sha256": sha256(check_path),
              "epoch": EPOCH, "new_physical_queries": 0, "max_queries_per_policy_and_case": 2,
              "minimum_seedwise_gain_against_best_fixed_rule": MIN_GAIN, "by_dataset": groups,
              "decision": "test_learned_follow_up_on_new_pages" if passed(groups) else "close_w6_supervision_route"}
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done", len(cases), len(cases), detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0
