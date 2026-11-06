"""P1 rollouts for one pool (protocol research/protocols/P1_noise_diagnosis.md).

    python -m work3.p1_run --pool-file POOL.parquet --pool-id 1 --tokenizer MODEL_DIR \
        --gen-url http://127.0.0.1:8101 --model-name p1-solver \
        --search-url http://127.0.0.1:8100/retrieve --out-dir OUT [--tool-config CFG]

Writes OUT/pool{id}_questions.jsonl (one line per question: 5 originals, 2 replays
each), OUT/pool{id}_lucky.jsonl (16 extra originals per k = 1 question) and
OUT/pool{id}_state.json. Resumable: finished questions are skipped on restart.
Sampling: 500 valid rows with numpy seed 20261106; while the k = 1 or the
k in {3,4} group has fewer than 60 questions, 500 more from the remaining rows with
seeds 20261107, 20261108 (at most 1500).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

from work3 import p1_agent as A

N_ORIG = 5
N_REPLAY = 2
N_LUCKY = 16
BATCH = 500
BATCH_SEEDS = (20261106, 20261107, 20261108)
MIN_GROUP = 60
QUESTION_MARKER = "<answer> Beijing </answer>. Question: "


# ---------------------------------------------------------------- pool rows
def pool_rows(df) -> list[dict]:
    """Valid rows: non-empty question (after the solver template) and answer."""
    rows = []
    for idx in range(len(df)):
        prompt = list(df["prompt"].iloc[idx])
        content = str(prompt[0]["content"]) if prompt else ""
        question = content.split(QUESTION_MARKER, 1)[1] if QUESTION_MARKER in content else ""
        rm = df["reward_model"].iloc[idx]
        try:
            target = [str(t) for t in list(rm["ground_truth"]["target"]) if t is not None]
        except (KeyError, TypeError):
            target = []
        if not question.strip() or not target or not target[0].strip():
            continue
        rows.append({"qid": int(idx), "question": question, "answer": target,
                     "data_source": str(df["data_source"].iloc[idx]),
                     "messages": [A.message(str(m["role"]), str(m["content"])) for m in prompt]})
    return rows


def draw_batch(valid_qids: list[int], taken: set[int], seed: int, size: int = BATCH) -> list[int]:
    remaining = sorted(q for q in valid_qids if q not in taken)
    if not remaining:
        return []
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.array(remaining), size=min(size, len(remaining)), replace=False)
    return sorted(int(x) for x in pick)


def group_counts(records: list[dict]) -> tuple[int, int]:
    k1 = sum(1 for r in records if r["k"] == 1)
    k34 = sum(1 for r in records if r["k"] in (3, 4))
    return k1, k34


# ---------------------------------------------------------------- clients
class Clients:
    def __init__(self, gen_url: str, model_name: str, search_url: str, concurrency: int):
        import httpx

        self.httpx = httpx
        self.gen_url = gen_url.rstrip("/") + "/v1/completions"
        self.model_name = model_name
        self.search_url = search_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=120.0, read=1200.0, write=120.0,
                                                              pool=120.0),
                                        limits=httpx.Limits(max_connections=4096))
        self.search_sem = asyncio.Semaphore(120)   # upstream SearchTool rate_limit
        self.calls = {"generate": 0, "search": 0, "search_errors": 0}

    async def generate(self, prompt: str, max_new: int) -> tuple[str, str]:
        payload = {"model": self.model_name, "prompt": prompt, "max_tokens": max_new,
                   "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "temperature": A.TEMPERATURE, "top_p": A.TOP_P, "stream": False}
        for attempt in range(5):
            try:
                r = await self.client.post(self.gen_url, json=payload)
                body = r.json()
                choice = body["choices"][0]
                self.calls["generate"] += 1
                usage = body.get("usage") or {}
                self.calls["prompt_tokens"] = self.calls.get("prompt_tokens", 0) + int(usage.get("prompt_tokens") or 0)
                self.calls["completion_tokens"] = (self.calls.get("completion_tokens", 0)
                                                   + int(usage.get("completion_tokens") or 0))
                return choice["text"], str(choice.get("finish_reason") or "stop")
            except Exception:  # noqa: BLE001 - mirror upstream retry, then raise
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    async def search(self, arguments) -> tuple[str, list[list[str]] | None]:
        ql = arguments.get("query_list") if isinstance(arguments, dict) else None
        if not ql or not isinstance(ql, list):
            return json.dumps({"result": "Error: 'query_list' is missing, empty, or not a list in "
                                         "parameters."}), None
        payload = {"queries": ql, "topk": A.TOPK, "return_scores": True}
        err = None
        async with self.search_sem:
            for attempt in range(10):
                try:
                    r = await self.client.post(self.search_url, json=payload, timeout=30)
                    if r.status_code in (500, 502, 503, 504):
                        err = f"Server Error ({r.status_code})"
                        await asyncio.sleep(attempt + 1)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    err = None
                    break
                except (self.httpx.ConnectError, self.httpx.TimeoutException) as e:
                    err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(attempt + 1)
                except Exception as e:  # noqa: BLE001
                    err = f"API Request Error: {e}"
                    break
        self.calls["search"] += 1
        if err is not None:
            self.calls["search_errors"] += 1
            return json.dumps({"result": f"Search error: API Call Failed: {err}"}), None
        raw = data.get("result", [])
        if not raw:
            return A.format_results([]), []
        passages = [[d["document"]["contents"] for d in block] for block in raw]
        return A.format_results(passages), passages


# ---------------------------------------------------------------- per question
def replay_rng(pool_id: int, qid: int, i: int, j: int):
    return np.random.default_rng([20261106, pool_id, qid, i, j])


async def run_question(row: dict, pool_id: int, renderer: A.Renderer, cl: Clients) -> dict:
    origs = await asyncio.gather(*[A.run_episode(row["messages"], renderer, cl.generate, cl.search)
                                   for _ in range(N_ORIG)])
    scores = [ep.score(row["answer"]) for ep in origs]

    async def one_replay(i: int, j: int) -> dict:
        prep = A.replay_prefix(origs[i], replay_rng(pool_id, row["qid"], i, j))
        if prep is None:
            return {"replayable": False, "score": scores[i], "new_tool_calls": 0, "stats": None,
                    "finish": None, "episode": None}
        prefix, results, used, stats = prep
        ep = await A.run_episode(prefix, renderer, cl.generate, cl.search, turns_used=used,
                                 tool_results=results)
        return {"replayable": True, "score": ep.score(row["answer"]),
                "new_tool_calls": ep.new_tool_calls, "stats": stats, "finish": ep.finish,
                "episode": ep.to_json()}

    replays = await asyncio.gather(*[asyncio.gather(*[one_replay(i, j) for j in range(N_REPLAY)])
                                     for i in range(N_ORIG)])
    return {"pool_id": pool_id, "qid": row["qid"], "question": row["question"],
            "answer": row["answer"], "data_source": row["data_source"],
            "scores": scores, "k": int(sum(scores)),
            "originals": [ep.to_json() for ep in origs],
            "replays": [list(r) for r in replays]}


async def run_lucky(row: dict, renderer: A.Renderer, cl: Clients) -> dict:
    eps = await asyncio.gather(*[A.run_episode(row["messages"], renderer, cl.generate, cl.search)
                                 for _ in range(N_LUCKY)])
    return {"qid": row["qid"], "scores": [ep.score(row["answer"]) for ep in eps],
            "finishes": [ep.finish for ep in eps]}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_pool(args) -> dict:
    import pandas as pd
    from transformers import AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    qfile = out / f"pool{args.pool_id}_questions.jsonl"
    lfile = out / f"pool{args.pool_id}_lucky.jsonl"
    sfile = out / f"pool{args.pool_id}_state.json"

    rows = pool_rows(pd.read_parquet(args.pool_file))
    by_qid = {r["qid"]: r for r in rows}
    tools, schema_source = A.upstream_tool_schemas(args.tool_config)
    renderer = A.Renderer(AutoTokenizer.from_pretrained(args.tokenizer), tools)
    cl = Clients(args.gen_url, args.model_name, args.search_url, args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)

    done = {r["qid"]: r for r in read_jsonl(qfile)}
    lock = asyncio.Lock()

    async def guarded(row, fn, path, store):
        async with sem:
            rec = await fn(row)
        async with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            store[rec["qid"]] = rec
            if len(store) % 50 == 0:
                print(f"[p1 pool{args.pool_id}] {path.name}: {len(store)} done", flush=True)

    batches, taken = [], set()
    t0 = time.time()
    for b, seed in enumerate(BATCH_SEEDS):
        qids = draw_batch(list(by_qid), taken, seed)
        if not qids:
            break
        batches.append({"seed": seed, "qids": qids})
        taken.update(qids)
        todo = [by_qid[q] for q in qids if q not in done]
        print(f"[p1 pool{args.pool_id}] batch {b + 1}: {len(qids)} drawn, {len(todo)} to run", flush=True)
        await asyncio.gather(*[guarded(r, lambda row: run_question(row, args.pool_id, renderer, cl),
                                       qfile, done) for r in todo])
        k1, k34 = group_counts([done[q] for q in taken])
        print(f"[p1 pool{args.pool_id}] after batch {b + 1}: k=1 {k1}, k in 3-4 {k34}", flush=True)
        if k1 >= MIN_GROUP and k34 >= MIN_GROUP:
            break

    lucky_done = {r["qid"]: r for r in read_jsonl(lfile)}
    k1_rows = [by_qid[q] for q in sorted(taken) if done[q]["k"] == 1 and q not in lucky_done]
    await asyncio.gather(*[guarded(r, lambda row: run_lucky(row, renderer, cl), lfile, lucky_done)
                           for r in k1_rows])

    state = {"pool_id": args.pool_id, "pool_file": str(args.pool_file), "tokenizer": args.tokenizer,
             "valid_rows": len(rows), "batches": batches, "questions": len(taken),
             "group_counts": dict(zip(("k1", "k34"), group_counts([done[q] for q in taken]))),
             "tool_schema_source": schema_source, "calls": cl.calls,
             "seconds": round(time.time() - t0, 1)}
    sfile.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    await cl.client.aclose()
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool-file", required=True)
    ap.add_argument("--pool-id", type=int, required=True, choices=(1, 3))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gen-url", required=True)
    ap.add_argument("--model-name", default="p1-solver")
    ap.add_argument("--search-url", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tool-config")
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args(argv)
    print(json.dumps(asyncio.run(run_pool(args)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
