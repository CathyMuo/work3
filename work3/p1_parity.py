"""Check P1 prompts against upstream verl prompt construction (factory env only).

P1 re-renders the whole conversation with the chat template at every turn;
upstream Dr. Zero builds `input_ids` incrementally inside `AsyncRolloutRequest`
(the code touched by the pinned "fix rollout inconsistency" commit). For every
episode that ends in a generated assistant turn, this rebuilds the upstream
request from the same messages and compares the final-turn generation prompt
(text and token ids) with P1's rendering. Reported only; not a gate.

    python -m work3.p1_parity --tokenizer MODEL_DIR --tool-config CFG \
        --questions OUT/pool1_questions.jsonl [...] --out OUT/p1_parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

from work3 import p1_agent as A


def upstream_prompt_ids(tok, tool_schema, prefix: list[dict]) -> list[int]:
    from verl.tools.schemas import (OpenAIFunctionCallSchema, OpenAIFunctionToolCall,
                                    OpenAIFunctionToolSchema)
    from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum

    first = [{"role": m["role"], "content": m["content"]} for m in prefix[:1]]
    req = AsyncRolloutRequest(
        batch_data_id=0, rollout_offset=0, request_id=str(uuid4()),
        state=AsyncRolloutRequestStateEnum.PENDING, messages=first, multi_modal_data=None,
        tool_schemas=[OpenAIFunctionToolSchema.model_validate(tool_schema)],
        tools_kwargs={"search": {"create_kwargs": {"data_source": "search_zero", "ground_truth": "",
                                                   "question": ""}}},
        interaction_kwargs={}, input_ids=None, response_ids=None, attention_mask=None,
        response_attention_mask=None, response_position_ids=None, response_loss_mask=None,
        reward_scores={}, max_prompt_len=A.PROMPT_LEN, max_response_len=A.RESPONSE_LEN,
        max_model_len=A.MAX_MODEL_LEN, use_inference_chat_template=False,
        tokenization_sanity_check_mode="strict", processing_class=tok,
    )
    i = 1
    while i < len(prefix):
        m = prefix[i]
        if m["role"] == "assistant":
            req.get_generation_prompt_ids(tok)          # upstream generates before appending
            calls = None
            if m.get("tool_calls"):
                calls = [OpenAIFunctionToolCall(id=c["id"], function=OpenAIFunctionCallSchema(
                    name=c["function"]["name"], arguments=c["function"]["arguments"]))
                    for c in m["tool_calls"]]
            req.add_assistant_message(tok, m["content"], tool_calls=calls)
            i += 1
        elif m["role"] == "tool":
            group = []
            while i < len(prefix) and prefix[i]["role"] == "tool":
                group.append(prefix[i]["content"]); i += 1
            req.add_tool_response_messages(tok, group)
        else:
            raise ValueError(f"unexpected role {m['role']}")
    return list(req.get_generation_prompt_ids(tok))


def check_file(path: Path, tok, renderer: A.Renderer, tool_schema, limit: int | None) -> dict:
    res = {"file": str(path), "checked": 0, "text_equal": 0, "ids_equal": 0, "errors": 0,
           "examples": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            eps = list(rec["originals"]) + [r["episode"] for rs in rec["replays"] for r in rs
                                            if r.get("episode")]
            for ep in eps:
                fidx = ep.get("final_turn_index")
                if fidx is None:
                    continue
                prefix = ep["messages"][:fidx]
                ours = renderer.render(prefix, gen=True)
                res["checked"] += 1
                try:
                    ids = upstream_prompt_ids(tok, tool_schema, prefix)
                except Exception as e:  # noqa: BLE001
                    res["errors"] += 1
                    if len(res["examples"]) < 3:
                        res["examples"].append({"qid": rec["qid"], "error": repr(e)[:300]})
                    continue
                text = tok.decode(ids)
                ours_ids = tok(text=[ours])["input_ids"][0]
                res["text_equal"] += text == ours
                res["ids_equal"] += list(ours_ids) == list(ids)
                if text != ours and len(res["examples"]) < 3:
                    k = next((j for j, (a, b) in enumerate(zip(text, ours)) if a != b),
                             min(len(text), len(ours)))
                    res["examples"].append({"qid": rec["qid"], "first_diff_char": k,
                                            "upstream": text[max(0, k - 80):k + 80],
                                            "ours": ours[max(0, k - 80):k + 80]})
                if limit and res["checked"] >= limit:
                    return res
    return res


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tool-config", required=True)
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max episodes per file (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tools, source = A.upstream_tool_schemas(args.tool_config)
    renderer = A.Renderer(tok, tools)
    files = [check_file(Path(p), tok, renderer, tools[0], args.limit or None) for p in args.questions]
    out = {"tool_schema_source": source, "files": files,
           "checked": sum(f["checked"] for f in files),
           "text_equal": sum(f["text_equal"] for f in files),
           "ids_equal": sum(f["ids_equal"] for f in files),
           "errors": sum(f["errors"] for f in files)}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
