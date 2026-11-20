"""Replay page-disjoint P0 failures with live queries and a shared frozen reader."""

import gzip
import json
import time
from collections import Counter

from work3 import wiki_rsi_action_diagnosis as diagnosis
from work3 import wiki_rsi_action_overnight as overnight
from work3 import wiki_rsi_action_pilot as pilot
from work3 import wiki_rsi_candidates as candidates
from work3 import wiki_rsi_natural as natural
from work3 import wiki_rsi_natural_query as query_format
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_two_query as pilot_replay
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import sha256, title_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS, revision
from work3.wiki_rsi_progress import atomic_json

PROTOCOL = "wiki_rsi_w7_live_follow_up_v1"
PREPARED = "w7_live_prepared.json"
PREP_SUMMARY = "wiki_rsi_w7_live_prepare_summary.json"
SUMMARY = "wiki_rsi_w7_live_summary.json"
POSITIVE = tuple(f"{seed}-positive_only" for seed in overnight.SEEDS)
ARMS = ("first_only", "linked_context", "frozen_follow_up", *POSITIVE)
LIMIT = 192
BATCH = 8
ANSWER_TOKENS = 64


def prepare(args, run_dir, progress):
    from transformers import AutoTokenizer

    earlier = run_dir / pilot_replay.SUMMARY
    gate = replay.read_json(earlier)
    if (gate["protocol"] != pilot_replay.PROTOCOL or gate["decision"] != "test_learned_follow_up_on_new_pages"
            or gate["w6c_sha256"] != sha256(run_dir / diagnosis.SUMMARY) or not pilot_replay.passed(gate["by_dataset"])):
        raise ValueError("Use the qualified W7a replay")
    _, overnight_prep, _, workers, old_cases, _ = diagnosis.cached_inputs(run_dir)
    models = {arm: workers / f"{arm}-{pilot_replay.EPOCH}.pt" for arm in POSITIVE}
    train_pairs = run_dir / "w6_overnight_train_pairs.json"
    if sha256(train_pairs) != overnight_prep["train_pairs_sha256"]:
        raise ValueError("Use the training pages from the completed four-GPU run")
    training = replay.read_json(train_pairs)["records"]
    dump, test = replay.input_files(args)
    natural_prep = replay.read_json(run_dir / natural.PREP_SUMMARY)
    provenance = {key: sha256(path) for key, path in (("dump_sha256", dump), ("test_sha256", test),
                  ("references_sha256", args.references), ("manifest_sha256", run_dir / "w0/wiki_rsi_manifest.jsonl"))}
    if any(natural_prep["inputs"][key] != value for key, value in provenance.items()):
        raise ValueError("Use the original pinned P0 and W0c inputs")
    replay.verify_checkpoint(args, overnight_prep)
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    tests, _ = replay.test_questions(test)
    with gzip.open(args.references, "rt", encoding="utf-8") as stream:
        refs = {ds: block["items"] for ds, block in json.load(stream)["datasets"].items()}
    excluded, _ = pilot.blocked_history(run_dir, refs)
    excluded.update(title_key(title) for case in old_cases for title in case["support_titles"])
    for row in training:
        excluded.update(title_key(row[key]) for key in ("source", "target", "foil"))
        excluded.update(title_key(title) for key in ("visible", "positive_return", "negative_return")
                        for title, _ in row[key])
    raw, counts = candidates.build_pool(dump, tests, refs, excluded)
    cases = candidates.choose_states(tok, test, natural.select_pool(raw, counts), counts, excluded, limit=LIMIT)
    if not all(counts[dataset]["selected"] for dataset in SUPPORT_DATASETS):
        raise ValueError(f"No new page-disjoint cases for one dataset. Pool counts: {counts}")
    corpus = args.corpus or args.drzero / "corpus/wiki-18.jsonl"
    catalog, digest, rows, _, _ = candidates.scan_corpus(corpus, cases, progress)
    if (digest, rows) != tuple(natural_prep["inputs"][key] for key in ("corpus_sha256", "corpus_rows")):
        raise ValueError("Use the original Wikipedia snapshot")
    for case in cases:
        question, visible = case["question"], case["visible"]
        rules, _ = natural.visible_queries(question, visible, catalog)
        case["linked_context"] = rules["linked_context"]
        case["aliases"] = tests[case["dataset"]][case["qid"]]
        case["title_stratum"] = ("literal" if pilot.literal_title(question, visible, case["probe_title"])
                                 else "not_literal")
    original = replay.read_json(run_dir / pilot.PREPARED)
    path = run_dir / PREPARED
    atomic_json(path, {"protocol": PROTOCOL, "cases": cases, "anchors": original["anchors"],
                       "retriever_id": original["retriever_id"], "worker_directory": workers.name})
    summary = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT), "prepared_sha256": sha256(path),
               "w7a_sha256": sha256(earlier), "model_sha256": overnight_prep["model_sha256"],
               "adapter_sha256": {arm: sha256(file) for arm, file in models.items()},
               "corpus_sha256": digest, "corpus_rows": rows, "pool": counts,
               "by_dataset_and_stratum": {ds: Counter(row["title_stratum"] for row in cases if row["dataset"] == ds)
                                          for ds in SUPPORT_DATASETS}}
    atomic_json(run_dir / PREP_SUMMARY, summary)
    progress.update("done", len(cases), len(cases), detail="ready")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def answer_prompt(tok, question, docs):
    return replay.render_ids(tok, [
        {"role": "system", "content": "Answer the question using the shown Wikipedia results. "
         "Return the shortest sufficient answer inside <answer> and </answer>. "
         "If the answer cannot be determined, return <answer>unknown</answer>."},
        {"role": "user", "content": f"Question: {question}\n\n"
         f"Observed Wikipedia results, as title and passage:\n{json.dumps(docs, ensure_ascii=False)}"},
    ], tools=())


def pack(tok, question, docs, render, max_new):
    items = list(dict.fromkeys(tuple(doc) for doc in docs))
    limit = replay.solver.MAX_MODEL_LEN - max_new - 1
    ids = render(tok, question, items)
    while len(ids) > limit and items:
        items.pop()
        ids = render(tok, question, items)
    if not ids or len(ids) > limit:
        raise ValueError("The base question does not fit the model context")
    return ids, items


def search_batch(retriever, wanted, first_queries=()):
    if first_queries:
        wanted = [query if query != first else None for query, first in zip(wanted, first_queries, strict=True)]
    unique = list(dict.fromkeys(query for query in wanted if query))
    returned = dict(zip(unique, retriever.search(unique), strict=True))
    return [(query, returned[query] if query else []) for query in wanted], len(unique)


def set_adapters(torch, adapters, weights=None):
    from work3 import wiki_rsi_action_model as model_tools

    if weights is not None:
        with torch.no_grad():
            for param, saved in zip(model_tools.trainable(adapters), weights, strict=True):
                if saved.shape != param.shape:
                    raise ValueError("Adapter shape differs from the original solver")
                param.copy_(saved)
    for adapter in adapters:
        adapter.enabled = weights is not None


def paired_measure(rows, arm, control, field):
    return replay.paired([{name: {"correct": int(row["arms"][name][field])} for name in (arm, control)}
                          for row in rows], arm, control)


def summarize(rows):
    report = {}
    for dataset in SUPPORT_DATASETS:
        selected = [row for row in rows if row["dataset"] == dataset]
        groups = {"all": selected, **{key: [row for row in selected if row["title_stratum"] == key]
                                     for key in pilot.EVAL_CAPS}}
        report[dataset] = {key: {"n": len(group), "arms": {
            arm: {field: sum(row["arms"][arm][field] for row in group)
                  for field in ("correct", "evidence_complete", "raw_evidence_complete", "queries", "valid")}
            for arm in ARMS},
            "learned_vs_controls": {arm: {control: {
                "strict_em": paired_measure(group, arm, control, "correct"),
                "packed_evidence": paired_measure(group, arm, control, "evidence_complete")}
                for control in ("linked_context", "frozen_follow_up")} for arm in POSITIVE}}
            for key, group in groups.items()}
    return report


def evaluate(cases, tok, model, adapters, weights, retriever, device, progress, torch, generate):
    rows, physical = [], Counter()

    def generate_queries(batch, histories):
        prompts = [pack(tok, case["question"], history, query_format.prompt_ids, query_format.MAX_NEW)[0]
                   for case, history in zip(batch, histories, strict=True)]
        generated = generate(model, tok, prompts, device, query_format.MAX_NEW)
        return [query_format.parse_query(item["text"], item["finish"])[0] for item in generated]

    for offset in range(0, len(cases), BATCH):
        batch = cases[offset:offset + BATCH]
        initial = [case["visible"] for case in batch]
        set_adapters(torch, adapters)
        first, calls = search_batch(retriever, generate_queries(batch, initial))
        physical["shared_first"] += calls
        histories = [[*docs, *result[1]] for docs, result in zip(initial, first, strict=True)]
        first_queries = [query for query, _ in first]
        outcomes = [{} for case in batch]
        for arm in ARMS:
            if arm == "first_only":
                wanted = [None] * len(batch)
            elif arm == "linked_context":
                wanted = [case["linked_context"] for case in batch]
            else:
                set_adapters(torch, adapters, weights.get(arm))
                wanted = generate_queries(batch, histories)
            second, calls = search_batch(retriever, wanted, first_queries)
            physical[arm] += calls
            final = [[*history, *result[1]] for history, result in zip(histories, second, strict=True)]
            inputs = [pack(tok, case["question"], docs, answer_prompt, ANSWER_TOKENS)
                      for case, docs in zip(batch, final, strict=True)]
            set_adapters(torch, adapters)
            answers = generate(model, tok, [ids for ids, _ in inputs], device, ANSWER_TOKENS)
            for index, (case, answer) in enumerate(zip(batch, answers, strict=True)):
                raw = final[index]
                visible = inputs[index][1]
                required = case["missing_facts"]
                outcomes[index][arm] = {**replay.score_generation(answer["text"], case["aliases"], answer["finish"]),
                    "evidence_complete": wiki.measure(visible, required)["all_missing_facts"],
                    "raw_evidence_complete": wiki.measure(raw, required)["all_missing_facts"],
                    "queries": int(bool(first_queries[index])) + int(bool(second[index][0])),
                    "second_query": second[index][0], "second_passages": second[index][1]}
        rows.extend({"dataset": case["dataset"], "qid": case["qid"], "title_stratum": case["title_stratum"],
                     "first_query": query, "first_passages": docs, "arms": arms}
                    for case, (query, docs), arms in zip(batch, first, outcomes, strict=True))
        progress.update("running", len(rows), len(cases), detail="evaluating_live_follow_up")
    return rows, physical


def run(args, run_dir, progress):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from work3 import wiki_rsi_action_model as model_tools

    start = time.monotonic()
    data, ready = replay.read_json(run_dir / PREPARED), replay.read_json(run_dir / PREP_SUMMARY)
    if (data["protocol"] != PROTOCOL or ready["protocol"] != PROTOCOL
            or ready["prepared_sha256"] != sha256(run_dir / PREPARED)
            or ready["w7a_sha256"] != sha256(run_dir / pilot_replay.SUMMARY)
            or data["retriever_id"] != args.retriever_id):
        raise ValueError("Use the original recorded preparation and index")
    replay.verify_checkpoint(args, ready)
    folder = run_dir / data["worker_directory"]
    if not folder.name.startswith("w6_overnight_workers_") or folder.parent != run_dir or folder.is_symlink():
        raise ValueError("Use the original W6b worker directory")
    models = {arm: folder / f"{arm}-{pilot_replay.EPOCH}.pt" for arm in POSITIVE}
    if any(sha256(path) != ready["adapter_sha256"][arm] for arm, path in models.items()):
        raise ValueError("W6b adapter weights changed")
    if torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one free H800")
    retriever = wiki.Retriever(args.retriever)
    passed, anchors = wiki.check_anchors(retriever, data["anchors"])
    if not passed:
        progress.update("blocked", detail="retriever_parity")
        print(json.dumps({"passed": passed, "by_dataset": anchors}, indent=2), flush=True)
        return 2
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), local_files_only=True, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to(device)
    weights = {arm: torch.load(path, map_location="cpu", weights_only=True) for arm, path in models.items()}
    probes, counts = evaluate(data["cases"], tok, model, model_tools.attach_adapters(model), weights, retriever,
                              device, progress, torch, model_tools.generate_queries)
    atomic_json(run_dir / "w7_live_cases.json", {"protocol": PROTOCOL, "records": probes})
    report = {"protocol": PROTOCOL, "repo_revision": revision(wiki.ROOT), "retriever_id": args.retriever_id,
              "preparation_sha256": sha256(run_dir / PREP_SUMMARY), "by_dataset_and_stratum": summarize(probes),
              "physical_queries_excluding_anchors": counts, "wall_seconds": round(time.monotonic() - start, 3)}
    atomic_json(run_dir / SUMMARY, report)
    progress.update("done", len(probes), len(probes), detail="completed")
    print(json.dumps(report, indent=2), flush=True)
    return 0
