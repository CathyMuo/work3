"""GPU workers of the plan-B first experiment: exact E5 search over the stored wiki-18 shards (never re-encoded) and
vLLM verification of claims against retrieved passages (yes / no probability from first-token log-probabilities).

Frozen protocol: research/protocols/claim_cert_b0.md. Batch sizes are set to the largest that fit, at the user's
request (2026-11-28): vLLM packs up to `max_batched_tokens` prompt tokens per step.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from work3.answer_prior_encode import Encoder, shard_docs
from work3.answer_prior_text import split_contents
from work3.wiki_rsi_progress import atomic_json

TOPK = 20
DOC_BLOCK = 1 << 21
QUERY_BLOCK = 4096
PASSAGE_CHARS = 1200
MAX_MODEL_LEN = 8192
PROMPT = ("You are checking a short answer against retrieved Wikipedia passages.\n\nPassages:\n{passages}\n\n"
          "Question: {question}\nProposed answer: {answer}\n\n"
          "Based only on the passages above, is the proposed answer correct? Reply with Yes or No.")
YES = ("Yes", "yes", " Yes", " yes", "YES")
NO = ("No", "no", " No", " no", "NO")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- search over stored shards

def search_worker(args):
    """Top-20 of every query over this rank's shards; refuses any shard whose manifest does not match."""
    import torch

    plan = load(args.plan)
    rank = args.rank
    status = Path(plan["directory"]) / f"status-search-{rank}.json"
    assigned = plan["assignments"][rank]
    atomic_json(status, {"rank": rank, "done": 0, "total": len(assigned), "updated_at": int(time.time())})
    encoder = Encoder("e5", plan["model"], dtype="float16", max_tokens=256)
    queries = load(plan["queries"])
    query_vectors = encoder.encode(queries, query=True, budget=262144)
    best_scores = torch.full((len(queries), TOPK), -float("inf"), device="cuda")
    best_ids = torch.full((len(queries), TOPK), -1, dtype=torch.long, device="cuda")
    for done, shard in enumerate(assigned, 1):
        manifest = load(shard["manifest"])
        expected = plan["expected_manifest"]
        if not manifest.get("complete") or any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{shard['manifest']} does not match the stored-index manifest {expected}")
        docs = shard_docs(manifest["docs"])
        stored = np.load(shard["vectors"], mmap_mode="r")
        if stored.shape != (len(docs), manifest["dim"]):
            raise ValueError(f"{shard['vectors']} has shape {stored.shape}, expected {(len(docs), manifest['dim'])}")
        doc_ids = torch.from_numpy(docs).to("cuda")
        for start in range(0, len(docs), DOC_BLOCK):
            block = torch.from_numpy(np.array(stored[start:start + DOC_BLOCK])).to("cuda").float()
            positions = doc_ids[start:start + len(block)]
            for first in range(0, len(queries), QUERY_BLOCK):
                scores = query_vectors[first:first + QUERY_BLOCK] @ block.T
                top_scores, top_index = scores.topk(min(TOPK, scores.shape[1]), dim=1)
                merged_scores = torch.cat([best_scores[first:first + QUERY_BLOCK], top_scores], dim=1)
                merged_ids = torch.cat([best_ids[first:first + QUERY_BLOCK], positions[top_index]], dim=1)
                keep_scores, keep = merged_scores.topk(TOPK, dim=1)
                best_scores[first:first + QUERY_BLOCK] = keep_scores
                best_ids[first:first + QUERY_BLOCK] = merged_ids.gather(1, keep)
                del scores, top_scores, top_index, merged_scores, merged_ids
            del block
        atomic_json(status, {"rank": rank, "done": done, "total": len(assigned), "updated_at": int(time.time())})
    np.savez(Path(plan["directory"]) / f"search-part-{rank}.npz", ids=best_ids.cpu().numpy(),
             scores=best_scores.cpu().numpy().astype(np.float32))


# ----------------------------------------------------------------------------- verification

def passage_block(contents, chars=PASSAGE_CHARS):
    lines = []
    for index, row in enumerate(contents, 1):
        title, text = split_contents(row)
        text = " ".join(text.split())[:chars]
        lines.append(f"[{index}] {title}: {text}" if title else f"[{index}] {text}")
    return "\n".join(lines)


def single_ids(tokenizer, variants):
    ids = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(encoded[0])
    return ids


def yes_probability(logprobs, yes_ids, no_ids):
    """Yes / (yes + no) over the first-token top log-probabilities; None when neither appears."""
    if not logprobs:
        return None
    first = logprobs[0]
    yes = sum(math.exp(item.logprob) for token, item in first.items() if token in yes_ids)
    no = sum(math.exp(item.logprob) for token, item in first.items() if token in no_ids)
    return yes / (yes + no) if yes + no > 0 else None


def verify_worker(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from work3.answer_prior_corpus import Corpus

    plan = load(args.plan)
    rank, ranks, tag = args.rank, plan["ranks"], plan["tag"]
    claims = [claim for index, claim in enumerate(load(plan["claims"])) if index % ranks == rank]
    with np.load(plan["search"]) as data:  # load once: NpzFile re-reads an array on every access
        search = {key: data[key] for key in ("q_ids", "qa_ids")}
    status = Path(plan["directory"]) / f"status-{tag}-{rank}.json"
    atomic_json(status, {"rank": rank, "done": 0, "total": len(claims), "updated_at": int(time.time())})
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True, use_fast=True)
    yes_ids, no_ids = single_ids(tokenizer, YES), single_ids(tokenizer, NO)
    if not yes_ids or not no_ids:
        raise ValueError("tokenizer has no single-token Yes / No")
    corpus = Corpus(plan["corpus"], plan["offsets"])

    def prompt(claim, passages_of, kind, k, chars=PASSAGE_CHARS):
        passages = passage_block(passages_of[kind][:k], chars)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(passages=passages, question=claim["question"],
                                                      answer=claim["answer"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    engine = LLM(model=plan["model"], tokenizer=plan["model"], dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
                 gpu_memory_utilization=plan["gpu_memory"], tensor_parallel_size=plan["tp"],
                 max_num_seqs=plan["max_num_seqs"], max_num_batched_tokens=plan["max_batched_tokens"],
                 enable_prefix_caching=True, seed=20261128 + rank, generation_config="vllm")
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    configs = [(key, "qa" if key.startswith("qa") else "q", int(key[2:] if key.startswith("qa") else key[1:]))
               for key in plan["configs"]]
    truncated = 0
    with (Path(plan["directory"]) / f"{tag}-{rank}.jsonl").open("x", encoding="utf-8") as stream:
        for start in range(0, len(claims), plan["chunk"]):
            chunk = claims[start:start + plan["chunk"]]
            prompts, owners = [], []
            for claim in chunk:
                # each claim's top-20 passages are read once; passages come first in the prompt, so a claim's
                # prompts for k = 1, 3, 5, 10, 20 share a cached prefix
                passages_of = {kind: [corpus.get(int(doc)) for doc in search[f"{kind}_ids"][claim["row"]][:TOPK]
                                      if doc >= 0] for kind in ("q", "qa")}
                for key, kind, k in configs:
                    ids = prompt(claim, passages_of, kind, k)
                    if len(ids) >= MAX_MODEL_LEN:
                        ids = prompt(claim, passages_of, kind, k, chars=PASSAGE_CHARS // 3)
                        truncated += 1
                    prompts.append({"prompt_token_ids": ids[-(MAX_MODEL_LEN - 2):]})
                    owners.append((claim["qid"], key, len(ids)))
            outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
            rows = {}
            for (qid, key, length), output in zip(owners, outputs):
                score = yes_probability(output.outputs[0].logprobs, yes_ids, no_ids)
                row = rows.setdefault(qid, {"qid": qid, "scores": {}, "parsed": {}, "tokens": {}})
                row["scores"][key] = 0.5 if score is None else score
                row["parsed"][key] = score is not None
                row["tokens"][key] = length
            for claim in chunk:
                stream.write(json.dumps(rows[claim["qid"]], ensure_ascii=False) + "\n")
            stream.flush()
            atomic_json(status, {"rank": rank, "done": start + len(chunk), "total": len(claims),
                                 "truncated": truncated, "updated_at": int(time.time())})
    corpus.close()
