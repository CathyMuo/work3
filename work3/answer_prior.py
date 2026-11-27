"""Answer-prior kill test (plan A, week 1): do dense retrievers rank passages by the answer a model already believes,
and does that hurt where the model is confidently wrong?

Frozen protocol: research/protocols/answer_prior_p0.md. Factory guide: docs/factory/answer_prior_p0.md.
Phases: verify (CPU), prepare (CPU corpus scan), smoke and run (4 GPUs), reanalyze (CPU), worker (internal).
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path
from time import monotonic, sleep, time

from work3.answer_prior_text import (PROTOCOL, alias_regex, alias_token_strings, choose_replacement, digest,
                                     has_answer, leaks, load_popqa, mentions, norm_answer, pair_facts, related,
                                     replacement_pool, split_contents, swap_item, swapped, title_key, token_string)
from work3.wiki_rsi_progress import REPO, Progress, atomic_json, progress_root

PHASE, PREPARE_PHASE = "answer_prior", "answer_prior_prepare"
SUMMARY = "wiki_rsi_answer_prior_summary.json"
PREPARE_SUMMARY = "wiki_rsi_answer_prior_prepare_summary.json"
PREPARE_ID = "answer-prior-prepare"
RUN_ID = re.compile(r"answer-prior-[a-z0-9][a-z0-9-]{0,48}")
EXPECTED_ROWS = 21015324
MAX_GPUS = 4
RETRIEVER_NAMES = ("e5", "qwen3")
SYSTEMS = ("e5", "qwen3", "bm25", "e5+bm25", "qwen3+bm25")
POOL = 20
SMOKE_PER_PROP, SMOKE_PREFIX_ROWS = 10, 200000
PARITY_N = 512
LIMITS = {"smoke": 2 * 3600, "run": 10 * 3600}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stop(signum, _frame):
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def sha256_file(path):
    digest_ = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest_.update(block)
    return digest_.hexdigest()


def model_inputs(model):
    model = Path(model).resolve(strict=True)
    weights = sorted(model.glob("*.safetensors"))
    if not weights or not (model / "config.json").is_file():
        raise ValueError(f"expected a local safetensors model with config.json: {model}")
    files = weights + sorted(model.glob("*.json")) + sorted(model.glob("*.jinja")) + sorted(model.glob("*.txt"))
    return str(model), {path.name: sha256_file(path) for path in files}


def versions():
    found = {}
    for package in ("torch", "transformers", "tokenizers", "vllm", "numpy", "pyarrow"):
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return found


def visible_gpus(requested):
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [value.strip() for value in devices.split(",") if value.strip()] if devices else [str(i) for i in range(requested)]
    if len(ids) < requested:
        raise RuntimeError(f"--gpus {requested} but only {len(ids)} visible devices: {devices!r}")
    return ids[:requested]


# ----------------------------------------------------------------------------- verify and prepare (CPU)

def verify(args):
    rows, files, dropped = load_popqa(args.popqa)
    props = Counter(row["prop"] for row in rows)
    corpus = Path(args.corpus).resolve(strict=True)
    with corpus.open("rb") as stream:
        first = json.loads(stream.readline())
    title, _ = split_contents(first.get("contents", ""))
    report = {"protocol": PROTOCOL, "popqa_rows": len(rows), "popqa_files": {name: sha256_file(path)
                                                                            for name, path in files.items()},
              "popqa_dropped": dropped, "relations": dict(sorted(props.items())),
              "with_s_pop": sum(row["s_pop"] is not None for row in rows),
              "corpus": str(corpus), "corpus_bytes": corpus.stat().st_size,
              "corpus_first_row": {"id": first.get("id"), "title": title}, "versions": versions()}
    for name in ("generator", "e5", "qwen3_embed"):
        path = getattr(args, name)
        if path is not None:
            model = Path(path).resolve(strict=True)
            report[name] = {"path": str(model), "config": (model / "config.json").is_file(),
                            "safetensors": len(list(model.glob("*.safetensors")))}
    if args.e5_index is not None:
        from work3.answer_prior_corpus import flat_index_vectors

        vectors = flat_index_vectors(args.e5_index)
        report["e5_index"] = None if vectors is None else {"rows": int(vectors.shape[0]), "dim": int(vectors.shape[1])}
    return report


def prepare(args):
    from work3.answer_prior_corpus import bm25_terms, scan_corpus

    directory = progress_root() / PREPARE_ID
    if (directory / PREPARE_SUMMARY).exists():
        raise FileExistsError(f"{directory / PREPARE_SUMMARY} exists; prepare runs once per corpus")
    directory.mkdir(parents=True, exist_ok=True)
    progress = Progress(directory, PREPARE_PHASE)
    started = monotonic()
    with progress.heartbeat():
        try:
            progress.update("running", detail="answer_prior_scanning")
            rows, files, dropped = load_popqa(args.popqa)
            vocabulary = sorted({word for row in rows for word in bm25_terms(row["question"])})
            titles = {title_key(row["s_wiki_title"]) for row in rows}
            corpus = scan_corpus(args.corpus, vocabulary, titles, directory, args.processes or min(32, os.cpu_count() or 1),
                                 lambda done, total, scanned: progress.update(
                                     "running", done, total, detail="answer_prior_scanning",
                                     metrics={"corpus_rows": scanned}))
            facts, subject_found = fact_passages(rows, directory / "subject_docs.jsonl")
            atomic_json(directory / "questions.json", rows)
            atomic_json(directory / "vocabulary.json", vocabulary)
            atomic_json(directory / "facts.json", facts)
            checks = {"rows_expected": corpus["rows"] == EXPECTED_ROWS,
                      "titled": corpus["untitled"] <= 0.01 * corpus["rows"]}
            summary = {"protocol": PROTOCOL, "repo_revision": progress.commit, "corpus": corpus,
                       "popqa": {"rows": len(rows), "files": {name: sha256_file(path) for name, path in files.items()},
                                 "dropped": dropped, "relations": dict(sorted(Counter(r["prop"] for r in rows).items()))},
                       "vocabulary": len(vocabulary), "subject_title_found": subject_found,
                       "fact_passages": len(facts), "checks": checks,
                       "verdict": "READY" if all(checks.values()) else "FAIL",
                       "wall_seconds": round(monotonic() - started, 1)}
            atomic_json(directory / PREPARE_SUMMARY, summary)
            progress.update("done" if summary["verdict"] == "READY" else "blocked", corpus["ranges"], corpus["ranges"],
                            detail=f"answer_prior_{summary['verdict'].lower()}", metrics={"corpus_rows": corpus["rows"]})
            return summary
        except (Exception, KeyboardInterrupt) as exc:
            atomic_json(directory / "incomplete.json", {"verdict": "INCOMPLETE", "phase": "prepare",
                                                        "repo_revision": progress.commit, "error": str(exc)})
            progress.update("failed", detail="answer_prior_incomplete")
            raise


def fact_passages(rows, subject_path):
    """Per question: the first passage of the subject page (by line number) whose text mentions an answer alias."""
    pages = defaultdict(list)
    with subject_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            title, _ = split_contents(item["contents"])
            pages[title_key(title)].append((item["doc"], item["contents"]))
    facts, found = {}, 0
    for row in rows:
        page = sorted(pages.get(title_key(row["s_wiki_title"]), []))
        found += bool(page)
        pattern = alias_regex(row["aliases"])
        for doc, contents in page:
            title, text = split_contents(contents)
            if mentions(text, pattern):
                facts[row["qid"]] = {"doc": doc, "contents": contents, "leak": leaks(row, title)}
                break
    return facts, found


# ----------------------------------------------------------------------------- worker processes

def spawn(kind, plan, rank, device, directory, parallel_tokenizer):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device,
                   "TOKENIZERS_PARALLELISM": "true" if parallel_tokenizer else "false"}
    command = [sys.executable, "-m", "work3.answer_prior", "worker", "--kind", kind, "--plan", str(plan),
               "--rank", str(rank)]
    log = (directory / f"worker-{kind}-{rank}.log").open("wb")
    try:
        return subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    finally:
        log.close()


def stop_workers(workers):
    for process in workers:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    for process in workers:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)


def wait_all(workers, deadline, progress, detail, status_glob, directory, extra=()):
    """Poll GPU workers (and optional background processes) until all exit; any failure stops everything."""
    every = list(workers) + list(extra)
    while True:
        codes = [process.poll() for process in every]
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError(f"{detail}: worker exit codes {codes} (None: still running, stopped after the failure);"
                               " inspect worker-*.log in the run directory")
        if all(code == 0 for code in codes[:len(workers)]):
            return
        states = []
        for path in sorted(directory.glob(status_glob)):
            with suppress(OSError, ValueError):
                states.append(load(path))
        done = sum(state.get("done", 0) for state in states)
        total = sum(state.get("total", 0) for state in states)
        progress.update("running", done, total, detail=detail,
                        workers=[{"name": f"rank-{state.get('rank', index)}", "device": str(state.get("rank", index)),
                                  "detail": detail, "done": state.get("done", 0), "total": state.get("total", 0),
                                  "metrics": {}, "updated_at": state.get("updated_at", int(time()))}
                                 for index, state in enumerate(states)],
                        metrics={"active_gpus": len(workers)})
        if monotonic() >= deadline:
            raise TimeoutError(f"{detail}: wall-clock limit reached")
        sleep(5)


def worker_main(args):
    from work3 import answer_prior_encode as encode

    if args.kind == "known":
        encode.known_worker(args)
    elif args.kind == "encode":
        encode.encode_worker(args)
    elif args.kind == "score":
        encode.score_worker(args)
    elif args.kind == "bm25":
        bm25_worker(args)
    else:
        raise ValueError(args.kind)


def bm25_worker(args):
    import numpy as np

    from work3.answer_prior_corpus import BM25

    plan = load(args.plan)
    index = BM25(plan["prepare"], load(Path(plan["prepare"]) / "vocabulary.json"))
    questions = load(plan["queries"])
    ids = np.full((len(questions), POOL * 5), -1, dtype=np.int64)
    scores = np.zeros((len(questions), POOL * 5), dtype=np.float32)
    for row, question in enumerate(questions):
        found, values = index.search(question, POOL * 5)
        ids[row, :len(found)], scores[row, :len(found)] = found, values
    np.savez(Path(plan["directory"]) / "search-bm25.npz", ids=ids, scores=scores)


# ----------------------------------------------------------------------------- experiment stages

def smoke_subset(rows):
    by_prop = defaultdict(list)
    for row in rows:
        by_prop[row["prop"]].append(row)
    return [row for prop in sorted(by_prop)
            for row in sorted(by_prop[prop], key=lambda row: digest("smoke", row["qid"]))[:SMOKE_PER_PROP]]


def knownness(context, questions, alias_norms):
    from work3.answer_prior_stats import stratum

    directory, gpus = context["directory"], context["gpus"]
    plan = directory / "plan-known.json"
    atomic_json(directory / "questions-known.json", [{"qid": r["qid"], "question": r["question"]} for r in questions])
    atomic_json(plan, {"model": context["generator"], "questions": str(directory / "questions-known.json"),
                       "gpus": len(gpus), "gpu_memory": context["gpu_memory"], "directory": str(directory)})
    workers = []
    try:
        workers = [spawn("known", plan, rank, device, directory, False) for rank, device in enumerate(gpus)]
        wait_all(workers, context["deadline"], context["progress"], "answer_prior_knownness", "status-known-*.json",
                 directory, extra=context["background"])
    finally:
        stop_workers(workers)
    answers = {}
    for rank in range(len(gpus)):
        with (directory / f"known-{rank}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                answers[item["qid"]] = item
    if set(answers) != {row["qid"] for row in questions}:
        raise ValueError(f"knownness returned {len(answers)} answers for {len(questions)} questions")
    result = {}
    for row in questions:
        item = answers[row["qid"]]
        result[row["qid"]] = stratum(item["greedy"], item["samples"], alias_norms[row["qid"]])
    return result


def dense_retrieval(context, questions, name):
    import numpy as np

    directory, gpus = context["directory"], context["gpus"]
    docs = context["docs"]
    shards = []
    root = context["index_root"] / name
    root.mkdir(parents=True, exist_ok=True)
    for rank in range(len(gpus)):
        if docs is None:
            rows = context["corpus_rows"]
            start, end = rows * rank // len(gpus), rows * (rank + 1) // len(gpus)
            spec = {"start": start, "end": end}
        else:
            path = directory / f"docs-{rank}.npy"
            if not path.exists():
                np.save(path, np.asarray(docs[rank::len(gpus)], dtype=np.int64))
            spec = {"file": str(path)}
        shards.append({"docs": spec, "vectors": str(root / f"shard-{rank}.npy"),
                       "manifest": str(root / f"shard-{rank}.json")})
    plan = directory / f"plan-encode-{name}.json"
    atomic_json(plan, {"retriever": name, "model": context["models"][name], "attention": context["attention"],
                       "corpus": context["corpus"], "offsets": context["offsets"], "queries": context["queries"],
                       "directory": str(directory), "shards": shards,
                       "manifest": {"protocol": PROTOCOL, "model_sha256": context["model_sha256"][name],
                                    "corpus_range_sha256": context["corpus_sha256"],
                                    "attention": context["attention"], "encoder_version": 1}})
    workers = []
    try:
        workers = [spawn("encode", plan, rank, device, directory, True) for rank, device in enumerate(gpus)]
        wait_all(workers, context["deadline"], context["progress"], f"answer_prior_encode_{name}",
                 f"status-{name}-*.json", directory, extra=context["background"])
    finally:
        stop_workers(workers)
    parts = [np.load(directory / f"search-{name}-{rank}.npz") for rank in range(len(gpus))]
    ids = np.concatenate([part["ids"] for part in parts], axis=1)
    scores = np.concatenate([part["scores"] for part in parts], axis=1)
    scores = np.where(ids >= 0, scores, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :POOL * 5]
    merged_ids, merged_scores = np.take_along_axis(ids, order, 1), np.take_along_axis(scores, order, 1)
    np.savez(directory / f"search-{name}.npz", ids=merged_ids, scores=merged_scores)
    stats = [load(directory / f"encode-{name}-{rank}.json") for rank in range(len(gpus))]
    return merged_ids, {"shards": stats, "norm_min": min(item["norm_min"] for item in stats),
                        "norm_max": max(item["norm_max"] for item in stats),
                        "fallbacks": sum(item["fallbacks"] for item in stats)}


def build_probes(context, questions, strata, search, contents_of):
    """Edited passages per retriever: T1 swaps on the retrieved fact passage, T3 cross-swaps, wrong-answer edits."""
    pools = context["replacement_pools"]
    facts = context["facts"]
    probe = {name: {"queries": [], "passages": [], "pairs": [], "t1": [], "t3": [], "attraction": [],
                    "reasons": Counter()} for name in RETRIEVER_NAMES}

    def add_query(store, text):
        store["queries"].append(text)
        return len(store["queries"]) - 1

    def add_pair(store, query, text):
        store["passages"].append(text)
        store["pairs"].append((query, len(store["passages"]) - 1))
        return len(store["pairs"]) - 1

    for name in RETRIEVER_NAMES:
        store = probe[name]
        ids = search[name]
        for index, row in enumerate(questions):
            top = [int(doc) for doc in ids[index][:POOL] if doc >= 0]
            key, pattern = title_key(row["s_wiki_title"]), alias_regex(row["aliases"])
            gold = None
            for doc in top[:5]:
                title, text = split_contents(contents_of[doc])
                if title is not None and title_key(title) == key and mentions(text, pattern):
                    gold = doc
                    break
            if gold is None:
                store["reasons"]["no_supported_passage_in_top5"] += 1
                continue
            replacement = choose_replacement(row, pools[row["prop"]], contents_of[gold])
            item, reason = swap_item(row, contents_of[gold], replacement)
            if item is None:
                store["reasons"][reason] += 1
                continue
            query = add_query(store, row["question"])
            pool = [add_pair(store, query, contents_of[doc]) for doc in top if doc != gold]
            store["t1"].append({"qid": row["qid"], "stratum": strata[row["qid"]]["stratum"], "pool": pool,
                                "original": add_pair(store, query, contents_of[gold]),
                                "treat": add_pair(store, query, item["treat"]),
                                "control": add_pair(store, query, item["control"]),
                                "answer_mentions": item["answer_mentions"],
                                "control_mentions": item["control_mentions"]})
    # T3: one retriever-independent set of passages, scored by every retriever
    fact_rows = []
    for row in questions:
        fact = facts.get(row["qid"])
        label = strata[row["qid"]]["stratum"]
        if fact and not fact["leak"] and label in ("known", "unknown", "wrong"):
            fact_rows.append({**row, "stratum": label, "contents": fact["contents"]})
    pairs = pair_facts(fact_rows)
    for name in RETRIEVER_NAMES:
        store = probe[name]
        for number, (a, b) in enumerate(pairs):
            queries = {fact["qid"]: add_query(store, fact["question"]) for fact in (a, b)}
            originals = {fact["qid"]: add_pair(store, queries[fact["qid"]], fact["contents"]) for fact in (a, b)}
            for fact, other in ((a, b), (b, a)):
                store["t3"].append({"pair": number, "qid": fact["qid"], "stratum": fact["stratum"],
                                    "original": originals[fact["qid"]],
                                    "edited": add_pair(store, queries[fact["qid"]],
                                                       swapped(fact["contents"], fact, other["obj"]))})
            if a["stratum"] != "wrong":
                continue
            for fact, other in ((a, b), (b, a)):
                answer = strata[fact["qid"]]["answer"]
                if len(answer) < 2 or related(answer, other["obj"]) or \
                        any(related(answer, alias) for alias in fact["aliases"] + other["aliases"]):
                    store["reasons"]["attraction_answer_unusable"] += 1
                    continue
                own, placebo = swapped(fact["contents"], fact, answer), swapped(other["contents"], other, answer)
                if own is None or placebo is None:
                    store["reasons"]["attraction_answer_present"] += 1
                    continue
                store["attraction"].append({"pair": number, "qid": fact["qid"],
                                            "own_original": originals[fact["qid"]],
                                            "own": add_pair(store, queries[fact["qid"]], own),
                                            "placebo_original": originals[other["qid"]],
                                            "placebo": add_pair(store, queries[other["qid"]], placebo)})
    return probe, {"facts_eligible": len(fact_rows), "pairs": len(pairs),
                   "pairs_by_stratum": dict(Counter(a["stratum"] for a, _ in pairs))}


def score_probes(context, probe, parity):
    import numpy as np

    directory, gpus = context["directory"], context["gpus"]
    workers, plans = [], {}
    try:
        for index, name in enumerate(RETRIEVER_NAMES):
            items = {key: probe[name][key] for key in ("queries", "passages", "pairs")}
            if name == "e5" and parity is not None:
                items["parity"] = {"stored": parity["stored"],
                                   "passages": list(range(len(items["passages"]), len(items["passages"]) +
                                                          len(parity["texts"])))}
                items["passages"] = items["passages"] + parity["texts"]
            atomic_json(directory / f"items-{name}.json", items)
            plans[name] = directory / f"plan-score-{name}.json"
            atomic_json(plans[name], {"retriever": name, "model": context["models"][name],
                                      "attention": context["attention"], "items": str(directory / f"items-{name}.json"),
                                      "directory": str(directory)})
            workers.append(spawn("score", plans[name], index, gpus[index % len(gpus)], directory, True))
        wait_all(workers, context["deadline"], context["progress"], "answer_prior_scoring", "status-score-*.json",
                 directory)
    finally:
        stop_workers(workers)
    scores = {name: np.load(directory / f"scores-{name}.npy") for name in RETRIEVER_NAMES}
    status = {name: load(directory / f"status-score-{name}.json") for name in RETRIEVER_NAMES}
    return scores, status


def parity_sample(context, contents_of_docs):
    """512 hash-chosen passages with their stored e5_Flat.index vectors, for an encoder parity readout only."""
    if context["e5_index"] is None:
        return None, "not_requested"
    from work3.answer_prior_corpus import flat_index_vectors

    vectors = flat_index_vectors(context["e5_index"])
    if vectors is None or vectors.shape[0] != context["corpus_rows"]:
        return None, "unparsable_or_row_mismatch"
    universe = context["docs_flat"] if context["docs_flat"] is not None else range(context["corpus_rows"])
    count = len(universe)
    chosen = sorted({int(universe[int(digest("parity", index), 16) % count]) for index in range(PARITY_N)})
    texts = contents_of_docs(chosen)
    return {"docs": chosen, "texts": [texts[doc] for doc in chosen],
            "stored": [vectors[doc].tolist() for doc in chosen]}, "ok"


def answer_hits(questions, search, contents_tokens, alias_tokens, k_values=(5, 10, 20)):
    hits = {}
    for system, ids in search.items():
        rows = []
        for index, row in enumerate(questions):
            top = [int(doc) for doc in ids[index][:max(k_values)] if doc >= 0]
            flags = [has_answer(contents_tokens[doc], alias_tokens[row["qid"]]) for doc in top]
            rows.append({k: any(flags[:k]) for k in k_values})
        hits[system] = rows
    return hits


def placebo_answers(questions, strata):
    """Wrong stratum: the next wrong question of the same relation (hash order) with an unrelated wrong answer."""
    by_prop = defaultdict(list)
    for row in questions:
        if strata[row["qid"]]["stratum"] == "wrong":
            by_prop[row["prop"]].append(row)
    placebo = {}
    for rows in by_prop.values():
        rows = sorted(rows, key=lambda row: digest("placebo", row["qid"]))
        for index, row in enumerate(rows):
            own = strata[row["qid"]]["answer"]
            for other in rows[index + 1:] + rows[:index]:
                answer = strata[other["qid"]]["answer"]
                if not related(answer, own) and not any(related(answer, alias) for alias in row["aliases"]):
                    placebo[row["qid"]] = answer
                    break
    return placebo


def analyse(context, questions, strata, search, probe, probe_meta, scores, score_status, retrieval_meta,
            contents_tokens):
    from work3.answer_prior_stats import (RECALL_FLOOR, RECALL_K, T2_K, attraction, decide, intrusion, pop_bin,
                                          pop_edges, preference, rounded, t1, t2, t3)

    alias_tokens = {row["qid"]: alias_token_strings(row["aliases"]) for row in questions}
    edges = pop_edges([row["s_pop"] for row in context["all_questions"]])
    hits = answer_hits(questions, search, contents_tokens, alias_tokens)
    placebo = placebo_answers(questions, strata)
    records, retrievers = [], {}
    for name in SYSTEMS:
        rows = []
        for index, row in enumerate(questions):
            item = {"qid": row["qid"], "stratum": strata[row["qid"]]["stratum"], "prop": row["prop"],
                    "pop_bin": pop_bin(row["s_pop"], edges), "hits": hits[name][index]}
            if item["stratum"] == "wrong" and row["qid"] in placebo:
                top = [int(doc) for doc in search[name][index][:T2_K] if doc >= 0]
                own_tokens = alias_token_strings([strata[row["qid"]]["answer"]])
                placebo_tokens = alias_token_strings([placebo[row["qid"]]])
                if own_tokens and placebo_tokens:
                    item["intrusion_own"] = float(any(has_answer(contents_tokens[d], own_tokens) for d in top))
                    item["intrusion_placebo"] = float(any(has_answer(contents_tokens[d], placebo_tokens) for d in top))
            rows.append(item)
        result = {"T2": t2(rows, gated=name in RETRIEVER_NAMES), "intrusion": intrusion(rows)}
        if name in RETRIEVER_NAMES:
            score = [float(value) for value in scores[name]]  # plain floats keep records JSON-serialisable
            t1_rows = []
            for item in probe[name]["t1"]:
                pool = [score[index] for index in item["pool"]]
                original, treat, control = score[item["original"]], score[item["treat"]], score[item["control"]]
                rank = lambda value: 1 + sum(other > value for other in pool)
                t1_rows.append({"qid": item["qid"], "stratum": item["stratum"], "rank_original": rank(original),
                                "rank_treat": rank(treat), "rank_control": rank(control),
                                "drop_treat": float(original - treat), "drop_control": float(original - control)})
            directions = [{"pair": item["pair"], "stratum": item["stratum"],
                           "pref": preference(score[item["original"]], score[item["edited"]])}
                          for item in probe[name]["t3"]]
            pulls = [{"pair": item["pair"], "own": preference(score[item["own_original"]], score[item["own"]]),
                      "placebo": preference(score[item["placebo_original"]], score[item["placebo"]])}
                     for item in probe[name]["attraction"]]
            recall_all = result["T2"]["recall"][str(RECALL_K)]["all"]
            result.update(T1=t1(t1_rows), T3=t3(directions), attraction=attraction(pulls),
                          valid=recall_all is not None and recall_all >= RECALL_FLOOR,
                          recall_at_20=recall_all, probe_reasons=dict(probe[name]["reasons"]),
                          retrieval=retrieval_meta[name], scoring=score_status[name])
            for item in t1_rows:
                item["retriever"] = name
            records.extend({"kind": "t1", **item} for item in t1_rows)
        retrievers[name] = result
    for index, row in enumerate(questions):
        info = strata[row["qid"]]
        records.append({"kind": "question", "qid": row["qid"], "prop": row["prop"], "stratum": info["stratum"],
                        "pop_bin": pop_bin(row["s_pop"], edges), "greedy_correct": info["greedy_correct"],
                        "samples_correct": info["samples_correct"], "mode_size": info["mode_size"],
                        "hits": {name: hits[name][index] for name in SYSTEMS}})
    counts = Counter(info["stratum"] for info in strata.values())
    generator = {"questions": len(strata), "accuracy": rounded(sum(info["greedy_correct"] for info in strata.values())
                                                                / max(1, len(strata))),
                 "abstain_rate": rounded(sum(info["abstains"] for info in strata.values()) / max(1, len(strata))),
                 "strata": dict(counts)}
    decision = decide({name: retrievers[name] for name in RETRIEVER_NAMES}, generator)
    return {"generator": generator, "retrievers": retrievers, "t3_pairs": probe_meta,
            "pop_edges": edges, **decision}, records


def experiment(args, directory, progress, started):
    import numpy as np

    from work3.answer_prior_corpus import Corpus

    deadline = started + LIMITS[args.phase]
    progress.update("running", detail="answer_prior_preflight")
    prepare_dir = progress_root() / PREPARE_ID
    prepared = load(prepare_dir / PREPARE_SUMMARY)
    if prepared.get("verdict") != "READY":
        raise ValueError(f"prepare verdict is {prepared.get('verdict')}; see {prepare_dir / PREPARE_SUMMARY}")
    corpus_path = Path(args.corpus).resolve(strict=True)
    if str(corpus_path) != prepared["corpus"]["path"] or corpus_path.stat().st_size != prepared["corpus"]["bytes"]:
        raise ValueError("--corpus differs from the prepared corpus")
    all_questions = load(prepare_dir / "questions.json")
    facts = load(prepare_dir / "facts.json")
    questions = smoke_subset(all_questions) if args.phase == "smoke" else all_questions
    models, model_sha256 = {}, {}
    for name, path in (("generator", args.generator), ("e5", args.e5), ("qwen3", args.qwen3_embed)):
        models[name], model_sha256[name] = model_inputs(path)
    found = versions()
    if found["vllm"] is None or found["torch"] is None or found["transformers"] is None:
        raise RuntimeError(f"vllm, torch and transformers are required; found {found}")
    inputs = {"protocol": PROTOCOL, "repo_revision": progress.commit, "model_sha256": model_sha256,
              "models": models, "corpus_range_sha256": prepared["corpus"]["range_sha256"],
              "popqa_files": prepared["popqa"]["files"], "versions": found, "attention": args.attention,
              "e5_index": str(args.e5_index) if args.e5_index else None}
    if args.phase == "run":
        smoke = load(progress_root() / args.smoke_run / SUMMARY)
        keys = ("protocol", "repo_revision", "model_sha256", "corpus_range_sha256", "popqa_files", "versions",
                "attention")
        if smoke.get("phase") != "smoke" or smoke.get("verdict") != "PASS" or \
                {key: smoke["provenance"].get(key) for key in keys} != {key: inputs[key] for key in keys}:
            raise ValueError("run requires a PASS smoke from this commit with the same models, corpus, data and runtime")
    gpus = visible_gpus(args.gpus)
    corpus = Corpus(corpus_path, prepare_dir / "offsets.npy")
    if len(corpus) != prepared["corpus"]["rows"]:
        raise ValueError("offsets do not match the prepared row count")
    docs = None
    if args.phase == "smoke":
        keys_needed = {title_key(row["s_wiki_title"]) for row in questions}
        subject = []
        with (prepare_dir / "subject_docs.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if title_key(split_contents(item["contents"])[0]) in keys_needed:
                    subject.append(item["doc"])
        docs = sorted(set(range(min(SMOKE_PREFIX_ROWS, len(corpus)))) | set(subject))
    atomic_json(directory / "queries.json", [row["question"] for row in questions])
    context = {"directory": directory, "gpus": gpus, "deadline": deadline, "progress": progress,
               "generator": models["generator"], "gpu_memory": args.gpu_memory, "models": models,
               "model_sha256": model_sha256, "attention": args.attention, "corpus": str(corpus_path),
               "offsets": str(prepare_dir / "offsets.npy"), "queries": str(directory / "queries.json"),
               "corpus_sha256": prepared["corpus"]["range_sha256"], "corpus_rows": len(corpus),
               "index_root": (args.index_root or progress_root() / "answer-prior-index") if docs is None
               else directory / "index", "docs": docs, "docs_flat": docs, "facts": facts,
               "replacement_pools": replacement_pool(all_questions), "e5_index": args.e5_index,
               "all_questions": all_questions, "background": []}
    atomic_json(directory / "inputs.json", {"provenance": inputs, "phase": args.phase, "questions": len(questions),
                                           "gpus": gpus, "smoke_docs": None if docs is None else len(docs)})
    alias_norms = {row["qid"]: [norm_answer(alias) for alias in row["aliases"]] for row in questions}
    bm25_plan = directory / "plan-bm25.json"
    atomic_json(bm25_plan, {"prepare": str(prepare_dir), "queries": str(directory / "queries.json"),
                            "directory": str(directory)})
    bm25 = spawn("bm25", bm25_plan, 0, "", directory, False)  # CPU only, overlaps the GPU stages
    context["background"] = [bm25]
    search, retrieval_meta = {}, {}
    try:
        strata = knownness(context, questions, alias_norms)
        atomic_json(directory / "knownness.json", strata)
        for name in RETRIEVER_NAMES:
            search[name], retrieval_meta[name] = dense_retrieval(context, questions, name)
        progress.update("running", detail="answer_prior_bm25")
        while bm25.poll() is None:
            if monotonic() >= deadline:
                raise TimeoutError("bm25: wall-clock limit reached")
            sleep(5)
        if bm25.returncode != 0:
            raise RuntimeError("bm25 worker failed; inspect worker-bm25-0.log")
    finally:
        stop_workers([bm25])
    bm25_ids = np.load(directory / "search-bm25.npz")["ids"]
    search["bm25"] = bm25_ids
    from work3.answer_prior_corpus import rrf

    for name in RETRIEVER_NAMES:
        fused = np.full((len(questions), POOL), -1, dtype=np.int64)
        for index in range(len(questions)):
            ranking = rrf([[d for d in search[name][index] if d >= 0], [d for d in bm25_ids[index] if d >= 0]], POOL)
            fused[index, :len(ranking)] = ranking
        search[f"{name}+bm25"] = fused
    progress.update("running", detail="answer_prior_probes")
    needed = {int(doc) for ids in search.values() for row in ids for doc in row[:POOL] if doc >= 0}
    contents_of = corpus.many(needed)
    probe, probe_meta = build_probes(context, questions, strata, search, contents_of)
    parity, parity_state = parity_sample(context, corpus.many)
    scores, score_status = score_probes(context, probe, parity)
    progress.update("running", detail="answer_prior_analyze")
    contents_tokens = {doc: token_string(text) for doc, text in contents_of.items()}
    result, records = analyse(context, questions, strata, search, probe, probe_meta, scores, score_status,
                              retrieval_meta, contents_tokens)
    corpus.close()
    with (directory / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    result.update(provenance=inputs, gpus=gpus, questions=len(questions), smoke_docs=None if docs is None else len(docs),
                  e5_parity={"state": parity_state, **(score_status["e5"].get("parity") or {})})
    if args.phase == "smoke":
        result["run_verdict_preview"] = result.pop("verdict")
        checks = {
            "generator_answered": sum(not info["abstains"] for info in strata.values()) >= 0.8 * len(strata),
            "unit_norm": all(0.98 <= retrieval_meta[name]["norm_min"] <= retrieval_meta[name]["norm_max"] <= 1.02
                             for name in RETRIEVER_NAMES),
            "search_filled": all(bool((search[name][:, 0] >= 0).all()) for name in RETRIEVER_NAMES),
            "bm25_filled": float((bm25_ids[:, 0] >= 0).mean()) >= 0.9,
            "t1_constructed": all(len(probe[name]["t1"]) >= 3 for name in RETRIEVER_NAMES),
            "t3_constructed": all(len(probe[name]["t3"]) >= 4 for name in RETRIEVER_NAMES),
        }
        result["smoke_checks"] = checks
        result["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return result


def run(args):
    if os.name != "posix":
        raise RuntimeError("run on the factory Linux environment")
    started = monotonic()
    directory = progress_root() / args.id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    progress = Progress(directory, PHASE)
    signal.signal(signal.SIGTERM, stop)
    try:
        with progress.heartbeat():
            result = experiment(args, directory, progress, started)
            result.update(phase=args.phase, wall_seconds=round(monotonic() - started, 1))
            atomic_json(directory / SUMMARY, result)
            verdict = result["verdict"].lower().replace("-", "_")
            progress.update("done" if verdict in ("pass", "go") else "blocked", 1, 1, detail=f"answer_prior_{verdict}",
                            metrics={"active_gpus": 0})
            print(json.dumps({key: result.get(key) for key in ("phase", "verdict", "passing_tests", "valid_retrievers",
                                                               "wall_seconds")}, ensure_ascii=False, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        atomic_json(directory / "incomplete.json", {"verdict": "INCOMPLETE", "phase": args.phase,
                                                    "repo_revision": progress.commit, "error": str(exc)})
        progress.update("failed", detail="answer_prior_incomplete")
        raise


def main():
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("verify", "prepare", "smoke", "run", "worker"))
    parser.add_argument("--id")
    parser.add_argument("--corpus", type=Path, help="wiki-18.jsonl")
    parser.add_argument("--popqa", type=Path, help="akariasai/PopQA directory or test.tsv")
    parser.add_argument("--generator", type=Path, help="Qwen3-4B (thinking switch off)")
    parser.add_argument("--e5", type=Path, help="intfloat/e5-base-v2")
    parser.add_argument("--qwen3-embed", type=Path, help="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--e5-index", type=Path, help="optional stored e5_Flat.index, parity readout only")
    parser.add_argument("--index-root", type=Path, help="where full-corpus embeddings are kept (default under "
                                                         "$WIKI_RSI_ROOT/answer-prior-index)")
    parser.add_argument("--smoke-run")
    parser.add_argument("--gpus", type=int, choices=range(1, MAX_GPUS + 1), default=MAX_GPUS)
    parser.add_argument("--gpu-memory", type=float, default=0.85, help="vLLM share for the knownness stage")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--processes", type=int, help="prepare: CPU processes (default min(32, cores))")
    parser.add_argument("--kind", choices=("known", "encode", "score", "bm25"), help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == "worker":
        worker_main(args)
        return
    if args.phase == "verify":
        if args.popqa is None or args.corpus is None:
            parser.error("verify requires --popqa and --corpus")
        print(json.dumps(verify(args), ensure_ascii=False, indent=2))
        return
    if args.phase == "prepare":
        if args.popqa is None or args.corpus is None:
            parser.error("prepare requires --popqa and --corpus")
        summary = prepare(args)
        print(json.dumps({key: summary[key] for key in ("verdict", "checks", "fact_passages", "subject_title_found",
                                                        "vocabulary", "wall_seconds")}, ensure_ascii=False, indent=2))
        print(json.dumps(summary["corpus"], ensure_ascii=False, indent=2))
        return
    if not RUN_ID.fullmatch(args.id or "") or args.id == PREPARE_ID:
        parser.error("--id must be answer-prior-<name> (lowercase letters, digits, dashes), not answer-prior-prepare")
    if None in (args.corpus, args.generator, args.e5, args.qwen3_embed):
        parser.error(f"{args.phase} requires --corpus, --generator, --e5 and --qwen3-embed")
    if args.phase == "run" and not RUN_ID.fullmatch(args.smoke_run or ""):
        parser.error("run requires --smoke-run answer-prior-<name> of a PASS smoke")
    if not 0.3 <= args.gpu_memory <= 0.95:
        parser.error("--gpu-memory must be within 0.3..0.95")
    run(args)


if __name__ == "__main__":
    main()
