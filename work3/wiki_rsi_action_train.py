"""Run the independent frozen / positive / paired one-query LoRA pilot."""

import json
import os
import re
import time
from collections import Counter

from work3 import wiki_rsi_action_pilot as w6a
from work3 import wiki_rsi_natural as w5a
from work3 import wiki_rsi_natural_query as w5b
from work3 import wiki_rsi_repair_audit as w4b
from work3 import wiki_rsi_repair_pool as pool
from work3 import wiki_rsi_replay as w1
from work3 import wiki_rsi_w2 as w2
from work3.evidence_ref import sha256, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w6b_independent_action_train_v1"
W6A_SUMMARY = "wiki_rsi_w6_action_pilot_summary.json"
TRAIN_PAIRS = "w6_action_train_pairs.json"
EVAL_BASELINES = "w6_action_eval_baselines.json"
PREPARED = "w6_action_train_prepared.json"
PREP_SUMMARY = "wiki_rsi_w6_action_train_prepare_summary.json"
ARMS = ("frozen", "positive_only", "paired")
MIN_NET_WINS = 5


def prior_inputs(run_dir):
    frozen = w1.read_json(run_dir / w6a.PREPARED)
    prepare = w1.read_json(run_dir / w6a.PREP_SUMMARY)
    summary = w1.read_json(run_dir / W6A_SUMMARY)
    training = w1.read_json(run_dir / TRAIN_PAIRS)
    evaluation = w1.read_json(run_dir / EVAL_BASELINES)
    if (any(item["protocol"] != w6a.PROTOCOL for item in (frozen, prepare, summary, training, evaluation))
            or summary["decision"] != "ready_for_independent_gpu_pilot"
            or prepare["prepared_sha256"] != sha256(run_dir / w6a.PREPARED)
            or summary["preparation_sha256"] != sha256(run_dir / w6a.PREP_SUMMARY)
            or summary["train_pairs_sha256"] != sha256(run_dir / TRAIN_PAIRS)
            or summary["evaluation_baselines_sha256"] != sha256(run_dir / EVAL_BASELINES)
            or not summary["retriever"]["passed"] or summary["retriever"]["retriever_id"] != frozen["retriever_id"]):
        raise ValueError("Use the qualified, unmodified W6a data and baselines")
    training, evaluation = training["records"], evaluation["records"]
    if (len(training) != summary["train_pairs"]
            or Counter(row["role"] for row in training) != summary["train_roles"]
            or w6a.eval_report(evaluation) != summary["evaluation_by_title_stratum"]
            or not w6a.readiness(training, summary["evaluation_by_title_stratum"])):
        raise ValueError("W6a case totals no longer match the accepted summary")
    eval_cases = frozen["evaluation"]
    old = {(row["dataset"], row["qid"]): row for row in evaluation}
    keys = {(case["dataset"], case["qid"]) for case in eval_cases}
    if len(old) != len(evaluation) or len(keys) != len(eval_cases) or keys != old.keys():
        raise ValueError("Evaluation case identities changed")
    for case in eval_cases:
        cached = old[case["dataset"], case["qid"]]
        returns = cached["returned"]
        if (set(returns) != set(w5a.case_queries(case))
                or {**w5a.score_case(case, returns), "title_stratum": case["title_stratum"]} != cached
                or w6a.literal_title(case["question"], case["visible"], case["probe_title"])
                   != (case["title_stratum"] == "literal")):
            raise ValueError("An independent evaluation baseline or stratum no longer reproduces")
    blocked = set(frozen["train_blocked"])
    blocked.update(title_key(title) for case in eval_cases
                   for title in [*case["support_titles"], *(name for name, _ in case["visible"])])
    used = set(blocked)
    for row in training:
        source, target, foil = (row[key] for key in ("source", "target", "foil"))
        pages = {title_key(name) for name in (source, target, foil)}
        evidence = w4b.birth_entries(target, row["positive_return"])["normalized"]
        if (len(pages) != 3 or pages & used or source not in frozen["train_sources"]
                or any(title_key(title) in blocked for title, _ in row["visible"])
                or row["positive_query"] != target or row["negative_query"] != foil
                or not evidence or evidence[0] != row["answer"]
                or pool.has_year(row["visible"], row["answer"])
                or pool.has_year([("", row["question"])], row["answer"])
                or pool.has_year(row["negative_return"], row["answer"])
                or not any(title_key(title) == title_key(foil) for title, _ in row["negative_return"])
                or any(title_key(title) == title_key(target) for title, _ in row["negative_return"])):
            raise ValueError("A training pair violates page isolation or no longer verifies")
        used.update(pages)
    return frozen, training


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    frozen, training = prior_inputs(run_dir)
    checkpoint = w1.read_json(run_dir / "wiki_rsi_prepare_summary.json")
    w1.verify_checkpoint(args, checkpoint)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError("Expected an end-of-sequence token in the original tokenizer")
    train_rows = []
    for row in training:
        ids = w5b.prompt_ids(tok, row["question"], row["visible"])
        pair = {key: tok(row[f"{key}_query"], add_special_tokens=False)["input_ids"] + [eos]
                for key in ("positive", "negative")}
        if not ids or any(not 1 < len(seq) <= w5b.MAX_NEW or len(ids) + len(seq) > w1.solver.MAX_MODEL_LEN
                          for seq in pair.values()):
            raise ValueError("A verified action pair does not fit the fixed token budget")
        train_rows.append({"prompt": ids, **pair})
    eval_rows, overflows = [], Counter()
    cases = frozen["evaluation"]
    for index, case in enumerate(cases, 1):
        ids = w5b.prompt_ids(tok, case["question"], case["visible"])
        overflow = len(ids) + w5b.MAX_NEW + 1 > w1.solver.MAX_MODEL_LEN
        overflows[case["dataset"]] += overflow
        eval_rows.append({key: case[key] for key in ("dataset", "qid", "title_stratum", "missing_facts")}
                         | {"ids": None if overflow else ids})
        progress.update("running", index, len(cases), detail="preparing_action_training")
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "train": train_rows, "evaluation": eval_rows,
                       "retriever_id": frozen["retriever_id"], "anchors": frozen["anchors"]})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "prepared_sha256": sha256(path),
               "model_sha256": checkpoint["model_sha256"], "w6a_summary_sha256": sha256(run_dir / W6A_SUMMARY),
               "train_pairs_sha256": sha256(run_dir / TRAIN_PAIRS), "eval_baselines_sha256": sha256(run_dir / EVAL_BASELINES),
               "train_pairs": len(train_rows), "evaluation_cases": len(eval_rows), "context_overflow": overflows}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(cases), len(cases), detail="ready",
                    metrics={"training_pairs": len(train_rows)})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def summarize(records):
    report = {}
    for dataset in SUPPORT_DATASETS:
        report[dataset] = {}
        for stratum in w6a.EVAL_CAPS:
            rows = [row["arms"] for row in records if row["dataset"] == dataset and row["title_stratum"] == stratum]
            report[dataset][stratum] = {
                "n": len(rows),
                "arms": {arm: {"all_missing_facts": sum(row[arm]["correct"] for row in rows),
                               "executed": sum(row[arm]["status"] == "executed" for row in rows)} for arm in ARMS},
                "paired_vs_positive": w1.paired(rows, "paired", "positive_only"),
                "paired_vs_frozen": w1.paired(rows, "paired", "frozen"),
                "positive_vs_frozen": w1.paired(rows, "positive_only", "frozen"),
            }
    passed = all(row["literal"][contrast]["wins"] - row["literal"][contrast]["losses"] >= MIN_NET_WINS
                 for row in report.values() for contrast in ("paired_vs_positive", "paired_vs_frozen"))
    return report, ("confirm_on_new_natural_pages" if passed else "no_go_for_current_paired_training")


def evaluate(model, tok, device, cases, retriever, progress, arm, generate_queries):
    results, query_count = [], 0
    for offset in range(0, len(cases), 8):
        batch = cases[offset:offset + 8]
        prompts = [case["ids"] for case in batch if case["ids"] is not None]
        responses = generate_queries(model, tok, prompts, device, w5b.MAX_NEW) if prompts else []
        if len(responses) != len(prompts):
            raise RuntimeError("Generation count changed")
        outputs = iter(responses)
        items = []
        for case in batch:
            output = next(outputs) if case["ids"] is not None else None
            query, status = w5b.parse_query(output["text"], output["finish"]) if output else (None, "context_overflow")
            items.append((query, status, output))
        queries = list(dict.fromkeys(query for query, _, _ in items if query))
        returned = dict(zip(queries, retriever.search(queries), strict=True))
        query_count += len(queries)
        for case, (query, status, output) in zip(batch, items, strict=True):
            docs = returned[query] if query else []
            coverage = w2.measure(docs, case["missing_facts"])
            results.append({"correct": int(coverage["all_missing_facts"]), "query": query,
                            "status": status, "output": output, "passages": docs})
        progress.update("running", len(results), len(cases), detail="evaluating_action_pilot",
                        metrics={"arm": arm, "queries": query_count})
    return results, query_count


def run(args, run_dir, progress):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from work3 import wiki_rsi_action_model as trainer

    path = run_dir / PREPARED
    data, ready = w1.read_json(path), w1.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or ready["protocol"] != PROTOCOL or ready["prepared_sha256"] != sha256(path)
            or data["retriever_id"] != args.retriever_id
            or ready["w6a_summary_sha256"] != sha256(run_dir / W6A_SUMMARY)
            or ready["train_pairs_sha256"] != sha256(run_dir / TRAIN_PAIRS)
            or ready["eval_baselines_sha256"] != sha256(run_dir / EVAL_BASELINES)
            or ready["train_pairs"] != len(data["train"]) or ready["evaluation_cases"] != len(data["evaluation"])):
        raise ValueError("Use the completed W6a run and matching W6b preparation")
    if not re.fullmatch(r"[0-9]+", os.environ.get("CUDA_VISIBLE_DEVICES", "")) or torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one idle physical GPU")
    w1.verify_checkpoint(args, ready)
    start = time.monotonic()
    retriever = w2.Retriever(args.retriever)
    passed, stats = w2.check_anchors(retriever, data["anchors"])
    check = {"protocol": PROTOCOL, "by_dataset": stats, "retriever_id": args.retriever_id, "passed": passed}
    atomic_json(run_dir / "wiki_rsi_w6_action_train_retriever_check.json", check)
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        print(json.dumps(check, indent=2), flush=True)
        return 2
    device = torch.device("cuda:0")
    progress.update("running", detail="loading_solver")
    torch.manual_seed(trainer.SEED)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), local_files_only=True,
                                                 torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if model.config.model_type != "qwen2":
        raise ValueError("Expected the frozen P0 Qwen2.5 model")
    adapters = trainer.attach_adapters(model)
    params = trainer.trainable(adapters)
    original = [param.detach().clone() for param in params]
    for adapter in adapters:
        adapter.enabled = False
    eval_rows, train_rows = data["evaluation"], data["train"]
    results, queries, budgets = {}, {}, {}
    results["frozen"], queries["frozen"] = evaluate(model, tok, device, eval_rows, retriever, progress,
                                                   "frozen", trainer.generate_queries)
    margins = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for index, row in enumerate(train_rows, 1):
            values = trainer.mean_logprobs(model, row["prompt"], (row["positive"], row["negative"]), pad, device)
            margins.append(float(values[0] - values[1]))
            progress.update("running", index, len(train_rows), detail="scoring_action_reference")
    for arm in ARMS[1:]:
        torch.manual_seed(trainer.SEED)
        for adapter in adapters:
            adapter.enabled = True
        with torch.no_grad():
            for param, initial in zip(params, original, strict=True):
                param.copy_(initial)
        budgets[arm] = trainer.train_arm(
            model, adapters, train_rows, margins, pad, device, paired=arm == "paired",
            progress=lambda done, total: progress.update("running", done, total, detail="training_action_pilot",
                                                          metrics={"arm": arm}),
        )
        results[arm], queries[arm] = evaluate(model, tok, device, eval_rows, retriever, progress, arm,
                                             trainer.generate_queries)
    for key in ("steps", "pair_presentations", "unpadded_input_tokens"):
        if budgets["positive_only"][key] != budgets["paired"][key]:
            raise RuntimeError("The two trained arms used different token or update budgets")
    rows = [{key: case[key] for key in ("dataset", "qid", "title_stratum")} |
            {"arms": {arm: results[arm][index] for arm in ARMS}} for index, case in enumerate(eval_rows)]
    report, decision = summarize(rows)
    atomic_json(run_dir / "w6_action_train_cases.json", {"protocol": PROTOCOL, "records": rows})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(w2.ROOT), "retriever": check,
               "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "by_dataset_and_title_stratum": report,
               "training": budgets, "physical_queries_excluding_anchors": queries,
               "maximum_new_tokens": w5b.MAX_NEW, "adapter_rank": trainer.RANK, "adapter_scale": trainer.SCALE,
               "trainable_parameters": sum(param.numel() for param in params),
               "epochs": trainer.EPOCHS, "learning_rate": trainer.LEARNING_RATE, "seed": trainer.SEED,
               "pair_weight": trainer.PAIR_WEIGHT, "min_net_wins_literal_per_dataset": MIN_NET_WINS,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": round(time.monotonic() - start, 3),
               "decision": decision}
    atomic_json(run_dir / "wiki_rsi_w6_action_train_summary.json", summary)
    progress.update("done", len(rows), len(eval_rows), detail="completed",
                    metrics={"training_pairs": len(train_rows), "queries": sum(queries.values())})
    print(json.dumps(summary, indent=2), flush=True)
    return 0
