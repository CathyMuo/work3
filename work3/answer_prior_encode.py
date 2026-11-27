"""GPU workers of the answer-prior kill test: closed-book knownness (vLLM), corpus encoding with exact search, and
fp32 scoring of edited passages. Each worker runs in its own process on one GPU (see answer_prior.py).

Frozen protocol: research/protocols/answer_prior_p0.md.
"""

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from work3.wiki_rsi_progress import atomic_json

QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"
RETRIEVERS = {
    # name: (kind, max tokens, corpus dtype, token budget per corpus batch)
    "e5": ("e5", 256, "float16", 262144),
    "qwen3": ("qwen3", 512, "float16", 131072),
}
TOPK = 100
CHUNK = 32768
DOC_BLOCK = 1 << 20
QUERY_BLOCK = 1024
KNOWN_SAMPLES, KNOWN_TEMPERATURE, KNOWN_TOP_P, KNOWN_MAX_TOKENS = 8, 0.7, 0.95, 24
KNOWN_PROMPT = ("Answer the question with only the answer itself (a name, place, title or short term). "
                "Do not explain.\n\nQuestion: {question}")
SEED = 20261127
KNOWN_CHUNK = 512


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_for(*parts):
    return int(hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------- encoders

class Encoder:
    """E5 (mean pooling, 'query: '/'passage: ') or Qwen3-Embedding (instructed queries, last token, explicit EOS)."""

    def __init__(self, kind, path, dtype="float16", attention="sdpa", max_tokens=None, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.kind, self.path, self.device = torch, kind, str(path), device
        self.max_tokens = max_tokens or RETRIEVERS[kind][1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, use_fast=True)
        self.dtype = getattr(torch, dtype)
        options = {"torch_dtype": self.dtype, "local_files_only": True}
        if kind == "qwen3":
            options["attn_implementation"] = attention
            self.eod = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if not isinstance(self.eod, int) or self.eod < 0:
                raise ValueError("Qwen3-Embedding tokenizer has no <|endoftext|> token")
            self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eod
        else:
            self.pad = self.tokenizer.pad_token_id or 0
        self.model = AutoModel.from_pretrained(self.path, **options).to(device).eval()
        self.fallback_model = None
        self.fallbacks = 0
        self.dim = int(self.model.config.hidden_size)

    def format(self, texts, query):
        if self.kind == "e5":
            return [("query: " if query else "passage: ") + text for text in texts]
        return [f"Instruct: {QWEN_TASK}\nQuery:{text}" for text in texts] if query else list(texts)

    def ids(self, texts, query=False):
        texts = self.format(texts, query)
        if self.kind == "e5":
            return self.tokenizer(texts, add_special_tokens=True, truncation=True, max_length=self.max_tokens)["input_ids"]
        encoded = self.tokenizer(texts, add_special_tokens=False, truncation=True,
                                 max_length=self.max_tokens - 1)["input_ids"]
        return [ids + [self.eod] for ids in encoded]

    def batches(self, ids, budget):
        """Length-sorted batches under a padded-token budget; yields index lists."""
        order = sorted(range(len(ids)), key=lambda index: len(ids[index]))
        batch, longest = [], 0
        for index in order:
            size = len(ids[index])
            if batch and max(longest, size) * (len(batch) + 1) > budget:
                yield batch
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, size)
        if batch:
            yield batch

    def forward(self, model, rows):
        torch = self.torch
        width = max(len(row) for row in rows)
        tokens = np.full((len(rows), width), self.pad, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for index, row in enumerate(rows):
            if self.kind == "qwen3":  # left padding: the last position is always the final real token
                tokens[index, width - len(row):] = row
                mask[index, width - len(row):] = 1
            else:
                tokens[index, :len(row)] = row
                mask[index, :len(row)] = 1
        tokens, mask = torch.from_numpy(tokens).to(self.device), torch.from_numpy(mask).to(self.device)
        hidden = model(input_ids=tokens, attention_mask=mask).last_hidden_state
        if self.kind == "qwen3":
            pooled = hidden[:, -1].float()
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * weights).sum(1) / weights.sum(1).clamp(min=1)).float()
        return torch.nn.functional.normalize(pooled, dim=-1)

    def fallback(self):
        if self.fallback_model is None:
            import copy

            self.fallback_model = copy.deepcopy(self.model).float().eval()
        return self.fallback_model

    def encode_ids(self, ids, budget, progress=None):
        torch = self.torch
        output = torch.empty((len(ids), self.dim), dtype=torch.float32, device=self.device)
        encoded = 0
        with torch.inference_mode():
            for batch in self.batches(ids, budget):
                vectors = self.forward(self.model, [ids[index] for index in batch])
                bad = ~torch.isfinite(vectors).all(dim=1)
                if bad.any():  # half-precision overflow on a rare passage: redo those rows in fp32
                    rows = [batch[index] for index in bad.nonzero().flatten().tolist()]
                    vectors[bad] = self.forward(self.fallback(), [ids[index] for index in rows])
                    self.fallbacks += len(rows)
                output[torch.tensor(batch, device=self.device)] = vectors
                encoded += len(batch)
                if progress is not None:
                    progress(encoded)
        return output

    def encode(self, texts, query=False, budget=65536, progress=None):
        return self.encode_ids(self.ids(texts, query), budget, progress)


# ----------------------------------------------------------------------------- corpus encoding and exact search

def shard_docs(spec):
    if "file" in spec:
        return np.load(spec["file"]).astype(np.int64)
    return np.arange(spec["start"], spec["end"], dtype=np.int64)


def encode_worker(args):
    """Encode one corpus shard (or reuse a complete one), then exact top-100 search of all queries over it."""
    import torch

    from work3.answer_prior_corpus import Corpus

    plan = load(args.plan)
    retriever, rank = plan["retriever"], args.rank
    shard = plan["shards"][rank]
    status = Path(plan["directory"]) / f"status-{retriever}-{rank}.json"
    kind, max_tokens, dtype, budget = RETRIEVERS[retriever]
    encoder = Encoder(kind, plan["model"], dtype=dtype, attention=plan["attention"], max_tokens=max_tokens)
    docs = shard_docs(shard["docs"])
    vectors_path, manifest_path = Path(shard["vectors"]), Path(shard["manifest"])
    expected = {**plan["manifest"], "rows": int(len(docs)), "dim": encoder.dim, "docs": shard["docs"]}
    reuse = manifest_path.is_file() and load(manifest_path) == {**expected, "complete": True}
    started = time.time()

    def report(done, stage):
        atomic_json(status, {"rank": rank, "retriever": retriever, "stage": stage, "done": int(done),
                             "total": int(len(docs)), "fallbacks": encoder.fallbacks, "reused": reuse,
                             "seconds": round(time.time() - started, 1), "updated_at": int(time.time())})

    if not reuse:
        manifest_path.unlink(missing_ok=True)
        corpus = Corpus(plan["corpus"], plan["offsets"])
        store = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=np.float16, shape=(len(docs), encoder.dim))
        prepared = queue.Queue(maxsize=2)

        def producer():
            try:
                for offset in range(0, len(docs), CHUNK):
                    block = docs[offset:offset + CHUNK]
                    texts = [corpus.get(int(doc)) for doc in block]
                    prepared.put((offset, encoder.ids(texts)))
                prepared.put(None)
            except BaseException as exc:  # surface reader/tokeniser errors in the main thread
                prepared.put(exc)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        done = 0
        report(0, "encoding")
        while True:
            item = prepared.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            offset, ids = item
            vectors = encoder.encode_ids(ids, budget)
            store[offset:offset + len(ids)] = vectors.to(torch.float16).cpu().numpy()
            done += len(ids)
            report(done, "encoding")
        thread.join()
        store.flush()
        del store
        corpus.close()
        atomic_json(manifest_path, {**expected, "complete": True})
    report(len(docs), "searching")
    queries = load(plan["queries"])
    query_vectors = encoder.encode(queries, query=True)
    stored = np.load(vectors_path, mmap_mode="r")
    if stored.shape != (len(docs), encoder.dim):
        raise ValueError(f"shard {vectors_path} has shape {stored.shape}, expected {(len(docs), encoder.dim)}")
    best_scores = torch.full((len(queries), TOPK), -float("inf"), device="cuda")
    best_ids = torch.full((len(queries), TOPK), -1, dtype=torch.long, device="cuda")
    norms = []
    for start in range(0, len(docs), DOC_BLOCK):
        block = torch.from_numpy(np.array(stored[start:start + DOC_BLOCK])).to("cuda").float()
        norms.append(block.norm(dim=1).cpu())
        positions = torch.arange(start, start + len(block), device="cuda")
        for first in range(0, len(queries), QUERY_BLOCK):
            scores = query_vectors[first:first + QUERY_BLOCK] @ block.T
            top_scores, top_index = scores.topk(min(TOPK, scores.shape[1]), dim=1)
            merged_scores = torch.cat([best_scores[first:first + QUERY_BLOCK], top_scores], dim=1)
            merged_ids = torch.cat([best_ids[first:first + QUERY_BLOCK], positions[top_index]], dim=1)
            keep_scores, keep = merged_scores.topk(TOPK, dim=1)
            best_scores[first:first + QUERY_BLOCK] = keep_scores
            best_ids[first:first + QUERY_BLOCK] = merged_ids.gather(1, keep)
        del block
    norms = torch.cat(norms)
    ids = best_ids.cpu().numpy()
    valid = ids >= 0
    global_ids = np.where(valid, docs[np.clip(ids, 0, None)], -1)
    np.savez(Path(plan["directory"]) / f"search-{retriever}-{rank}.npz", ids=global_ids,
             scores=best_scores.cpu().numpy().astype(np.float32))
    atomic_json(Path(plan["directory"]) / f"encode-{retriever}-{rank}.json", {
        "rank": rank, "rows": int(len(docs)), "dim": encoder.dim, "reused": reuse, "fallbacks": encoder.fallbacks,
        "norm_min": float(norms.min()), "norm_max": float(norms.max()), "seconds": round(time.time() - started, 1)})
    report(len(docs), "done")


# ----------------------------------------------------------------------------- fp32 scoring of edited passages

def score_worker(args):
    """Score (query, passage) pairs in fp32 for one retriever; every passage in a pool is encoded in this process."""
    plan = load(args.plan)
    retriever = plan["retriever"]
    kind, max_tokens, _, _ = RETRIEVERS[retriever]
    encoder = Encoder(kind, plan["model"], dtype="float32", attention=plan["attention"], max_tokens=max_tokens)
    items = load(plan["items"])
    status = Path(plan["directory"]) / f"status-score-{retriever}.json"
    total = len(items["queries"]) + len(items["passages"])
    last = [0.0]

    def report(done, force=False):
        if force or time.time() - last[0] >= 5:
            last[0] = time.time()
            atomic_json(status, {"rank": args.rank, "retriever": retriever, "stage": "encoding", "done": int(done),
                                 "total": total, "updated_at": int(time.time())})

    report(0, force=True)
    queries = encoder.encode(items["queries"], query=True, budget=32768, progress=report)
    offset = len(items["queries"])
    passages = encoder.encode(items["passages"], query=False, budget=32768, progress=lambda done: report(offset + done))
    import torch

    pairs = torch.as_tensor(np.asarray(items["pairs"], dtype=np.int64).reshape(-1, 2), device=queries.device)
    scores = (queries[pairs[:, 0]] * passages[pairs[:, 1]]).sum(dim=1).cpu().numpy().astype(np.float64)
    np.save(Path(plan["directory"]) / f"scores-{retriever}.npy", scores)
    parity = None
    if items.get("parity"):
        stored = np.asarray(items["parity"]["stored"], dtype=np.float32)
        fresh = passages[torch.tensor(items["parity"]["passages"], device=passages.device)].cpu().numpy()
        stored /= np.linalg.norm(stored, axis=1, keepdims=True)
        cosine = (stored * fresh).sum(axis=1)
        parity = {"n": int(len(cosine)), "mean_cosine": float(cosine.mean()), "min_cosine": float(cosine.min())}
    atomic_json(status, {"rank": args.rank, "retriever": retriever, "stage": "done", "done": total, "total": total,
                         "pairs": int(len(pairs)), "parity": parity, "fallbacks": encoder.fallbacks,
                         "updated_at": int(time.time())})


# ----------------------------------------------------------------------------- closed-book knownness (vLLM)

def known_worker(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    plan = load(args.plan)
    rank, gpus = args.rank, plan["gpus"]
    rows = [row for index, row in enumerate(load(plan["questions"])) if index % gpus == rank]
    tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True, use_fast=True)
    stops = sorted({value for value in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                        tokenizer.convert_tokens_to_ids("<|endoftext|>"))
                    if isinstance(value, int) and value >= 0})

    def prompt(question):
        messages = [{"role": "user", "content": KNOWN_PROMPT.format(question=question)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    engine = LLM(model=plan["model"], tokenizer=plan["model"], dtype="bfloat16", max_model_len=1024,
                 gpu_memory_utilization=plan["gpu_memory"], seed=SEED + rank, enable_prefix_caching=True,
                 generation_config="vllm")
    status = Path(plan["directory"]) / f"status-known-{rank}.json"
    atomic_json(status, {"rank": rank, "done": 0, "total": len(rows), "updated_at": int(time.time())})
    with (Path(plan["directory"]) / f"known-{rank}.jsonl").open("x", encoding="utf-8") as stream:
        # chunks of KNOWN_CHUNK questions (9 sequences each) keep vLLM batches full and give the board progress
        for start in range(0, len(rows), KNOWN_CHUNK):
            chunk = rows[start:start + KNOWN_CHUNK]
            prompts, params = [], []
            for row in chunk:
                ids = prompt(row["question"])
                prompts += [{"prompt_token_ids": ids}] * 2
                # no newline stop: a leading newline would end the answer empty; clean_answer keeps the first line
                params += [SamplingParams(n=1, temperature=0.0, max_tokens=KNOWN_MAX_TOKENS, stop_token_ids=stops,
                                          skip_special_tokens=True),
                           SamplingParams(n=KNOWN_SAMPLES, temperature=KNOWN_TEMPERATURE, top_p=KNOWN_TOP_P,
                                          max_tokens=KNOWN_MAX_TOKENS, stop_token_ids=stops, skip_special_tokens=True,
                                          seed=seed_for("known", row["qid"]))]
            outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
            for index, row in enumerate(chunk):
                greedy, sampled = outputs[2 * index], outputs[2 * index + 1]
                stream.write(json.dumps({"qid": row["qid"], "greedy": greedy.outputs[0].text,
                                         "samples": [choice.text for choice in sampled.outputs],
                                         "prompt_tokens": len(greedy.prompt_token_ids)}, ensure_ascii=False) + "\n")
            stream.flush()
            atomic_json(status, {"rank": rank, "done": start + len(chunk), "total": len(rows),
                                 "updated_at": int(time.time())})
