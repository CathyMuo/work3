"""Train search/read LoRA signals and run two rounds of matched-budget allocation."""

import copy
import time
from pathlib import Path
from random import Random

from work3 import wiki_rsi_feedback_data as data
from work3 import wiki_rsi_feedback_protocol as p
from work3 import wiki_rsi_natural_query as query
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_two_query_live as reader
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import sha256, text_key
from work3.wiki_rsi_progress import atomic_json


def capped_prompt(tok, question, docs, render):
    shown = list(dict.fromkeys(tuple(item) for item in docs))
    while True:
        ids = render(tok, question, shown)
        if len(ids) <= p.INPUT_LIMIT:
            return ids, shown
        if not shown:
            raise ValueError("Question prompt exceeds the frozen token limit")
        shown.pop()


def encode_pair(tok, row):
    question = row["question"]
    search, visible = capped_prompt(tok, question, row["first"], query.prompt_ids)
    read, final = capped_prompt(tok, question, [*row["first"], *row["second"]], reader.answer_prompt)
    if not (wiki.measure(visible, row["facts"][:1])["all_missing_facts"] and
            wiki.measure(final, row["facts"])["all_missing_facts"]):
        return None
    outputs = {"search": row["bridge"], "read": f"<answer>{row['answer']}</answer>"}
    prompts = {"search": search, "read": read}
    examples = {}
    for signal, completion in outputs.items():
        ids = list(tok(completion, add_special_tokens=False)["input_ids"]) + [tok.eos_token_id]
        if not (ids[0] is not None and
                len(ids) <= (p.QUERY_TOKENS if signal == "search" else p.ANSWER_TOKENS)):
            return None
        examples[signal] = (prompts[signal], ids)
    width = max(len(prompt) + len(completion) for prompt, completion in examples.values())
    return {"examples": examples, "width": width, "triage": row["triage"]}


def training_groups(n, seed, steps):
    needed = sum(steps) * p.ACCUMULATION
    if n < needed:
        raise ValueError("Too few distinct paired examples for the frozen exposures")
    order = Random(seed).sample(range(n), needed)
    groups = list(data.batches(order, p.ACCUMULATION))
    offset, parts = 0, []
    for count in steps:
        parts.append(groups[offset:offset + count])
        offset += count
    return parts


def fixed_signal(arm, index, row):
    if arm == "triage":
        return row["triage"]
    return "search" if index % 4 < p.FIXED[arm] else "read"


def evaluate(tok, model, device, cases, retriever, generate, progress, detail):
    scores, queries, tokens = {}, 0, 0
    for batch in data.batches(cases):
        prompts = [capped_prompt(tok, row["question"], row["first"], query.prompt_ids) for row in batch]
        results = generate(model, tok, [ids for ids, _ in prompts], device, p.QUERY_TOKENS)
        wanted = [query.parse_query(item["text"], item["finish"])[0] for item in results]
        wanted = [value if value and text_key(value) != text_key(row["question"]) else None
                  for row, value in zip(batch, wanted, strict=True)]
        returned, executed = reader.search_batch(retriever, wanted)
        tokens += sum(len(ids) + output["tokens"] for (ids, _), output in zip(prompts, results, strict=True))
        queries += executed
        answer_inputs = [capped_prompt(tok, row["question"], [*shown, *looked_up[1]], reader.answer_prompt)
                         for row, (_, shown), looked_up in zip(batch, prompts, returned, strict=True)]
        answers = generate(model, tok, [ids for ids, _ in answer_inputs], device, p.ANSWER_TOKENS)
        tokens += sum(len(ids) + output["tokens"] for (ids, _), output in zip(answer_inputs, answers, strict=True))
        for row, output, (_, visible), (executed_query, _) in zip(batch, answers, answer_inputs, returned, strict=True):
            score = replay.score_generation(output["text"], row["aliases"], output["finish"])
            scores[row["qid"]] = {"dataset": row["dataset"], "correct": score["correct"], "valid": score["valid"],
                                  "cluster": row["cluster"], "second_query": bool(executed_query),
                                  "evidence": wiki.measure(visible, row["facts"])["all_missing_facts"]}
        progress(detail, len(scores), len(cases), validation_cases=len(scores), queries=queries)
    return scores, {"queries": queries, "tokens": tokens}


class Training:
    def __init__(self, tok, model, device, seed, progress):
        import torch
        from work3 import wiki_rsi_action_model as adapters

        self.torch, self.generate = torch, adapters.generate_queries
        self.learning_rate = adapters.LEARNING_RATE
        self.tok, self.model, self.device, self.progress = tok, model, device, progress
        torch.manual_seed(seed)
        self.adapters = adapters.attach_adapters(model)
        self.params = adapters.trainable(self.adapters)
        self.optimizer = torch.optim.AdamW(self.params, lr=self.learning_rate, weight_decay=0)
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.steps, self.tokens, self.probe_tokens, self.probe_queries = 0, 0, 0, 0

    def snapshot(self):
        return [weight.detach().clone() for weight in self.params], copy.deepcopy(self.optimizer.state_dict())

    def restore(self, snapshot):
        weights, optim = snapshot
        with self.torch.no_grad():
            for param, value in zip(self.params, weights, strict=True):
                param.copy_(value)
        # A branch must never share mutable moment tensors with its parent or its sibling.
        self.optimizer.load_state_dict(copy.deepcopy(optim))

    def loss(self, row, signal):
        from torch.nn import functional as F

        prompt, target = row["examples"][signal]
        sequence = [*prompt, *target]
        width = row["width"]
        ids = self.torch.tensor([sequence + [self.pad] * (width - len(sequence))], device=self.device)
        mask = self.torch.arange(width, device=self.device)[None, :] < len(sequence)
        hidden = self.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        logits = self.model.get_output_embeddings()(hidden[0, len(prompt) - 1:len(sequence) - 1])
        labels = ids[0, len(prompt):len(sequence)]
        return F.cross_entropy(logits.float(), labels)

    def update(self, examples, groups, route, detail):
        self.model.train()
        for step, group in enumerate(groups, 1):
            self.optimizer.zero_grad(set_to_none=True)
            for index, number in enumerate(group):
                row = examples[number]
                signal = route(index, row)
                (self.loss(row, signal) / len(group)).backward()
                self.tokens += row["width"]
            self.torch.nn.utils.clip_grad_norm_(self.params, 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.steps += 1
            self.progress(detail, step, len(groups), train_steps=self.steps, prompt_tokens=self.tokens)

    def assess(self, cases, retriever, detail, *, probe=False):
        scores, cost = evaluate(self.tok, self.model, self.device, cases, retriever,
                                self.generate, self.progress, detail)
        if probe:
            self.probe_tokens += cost["tokens"]
            self.probe_queries += cost["queries"]
        return scores


def probe_round(trainer, examples, prefix, tail, schedule, retriever, frozen_choice=None):
    parent = trainer.snapshot()
    scores, children = {}, {}
    for signal in p.SIGNALS:
        trainer.restore(parent)
        trainer.update(examples, prefix, lambda _index, _row: signal, f"feedback_probe_{signal}")
        scores[signal] = trainer.assess(schedule, retriever, f"feedback_score_{signal}", probe=True)
        children[signal] = trainer.snapshot()
    allocation = p.probe_choice(scores["search"], scores["read"])
    choice = allocation["choice"] if frozen_choice is None else frozen_choice
    allocation["applied"] = choice
    trainer.restore(children.get(choice, parent))
    route = (lambda index, row: fixed_signal("fixed50", index, row)) if choice == "mix" else (
        lambda _index, _row: choice)
    trainer.update(examples, tail, route, "feedback_continuation")
    return allocation


def fixed_round(trainer, examples, prefix, tail, schedule, retriever, arm):
    def route(index, row):
        return fixed_signal(arm, index, row)

    for _ in range(2):
        trainer.update(examples, prefix, route, "feedback_fixed_probe_budget")
        trainer.assess(schedule, retriever, "feedback_score_fixed", probe=True)
    trainer.update(examples, tail, route, "feedback_continuation")


def run(spec, locked, pairs, tok, model, device, progress, out):
    phase, seed, arm = spec["phase"], spec["seed"], spec["arm"]
    retriever = wiki.Retriever(spec["retriever"])
    trainer = Training(tok, model, device, seed, progress)
    allocations = []
    if arm == "base":
        reader.set_adapters(trainer.torch, trainer.adapters)
        split = "confirmation" if phase == "confirm" else "development"
        return {"scores": trainer.assess(locked["splits"][split], retriever, f"feedback_{split}")}
    if phase == "confirm":
        weights = Path(spec["adapter_path"])
        if sha256(weights) != spec["weights_sha256"]:
            raise ValueError("Confirmation adapter differs from the committed pilot receipt")
        reader.set_adapters(trainer.torch, trainer.adapters,
                            trainer.torch.load(weights, map_location="cpu", weights_only=True))
        scores = trainer.assess(locked["splits"]["confirmation"], retriever, "feedback_confirm")
        return {"scores": scores, "training_seconds": spec["training_seconds"],
                "choices": spec["choices"], "weights_sha256": spec["weights_sha256"]}

    examples = [encode_pair(tok, row) for row in pairs]
    if any(row is None for row in examples):
        raise ValueError("Approved training examples do not fit the frozen input format")
    start = time.monotonic()
    if phase == "mechanism":
        groups, = training_groups(len(examples), seed, [p.MECHANISM_STEPS])
        trainer.update(examples, groups, lambda _index, _row: arm, f"feedback_probe_{arm}")
    else:
        all_groups = training_groups(len(examples), seed, [p.PROBE_STEPS, p.CONTINUE_STEPS] * p.ROUNDS)
        first_choice = None
        for prefix, tail in zip(all_groups[::2], all_groups[1::2], strict=True):
            schedule = locked["splits"]["schedule"]
            if arm in {"adaptive", "frozen_probe"}:
                applied = first_choice if arm == "frozen_probe" else None
                allocation = probe_round(trainer, examples, prefix, tail, schedule, retriever, applied)
                allocations.append(allocation)
                first_choice = allocations[0]["applied"]
            else:
                fixed_round(trainer, examples, prefix, tail, schedule, retriever, arm)
    trainer.torch.cuda.synchronize(device)
    seconds = round(time.monotonic() - start, 4)
    scores = trainer.assess(locked["splits"]["development"], retriever, "feedback_development")
    report = {"scores": scores, "training_seconds": seconds, "train_steps": trainer.steps,
              "padded_train_tokens": trainer.tokens, "probe_tokens": trainer.probe_tokens,
              "probe_queries": trainer.probe_queries, "allocations": allocations,
              "choices": [item["applied"] for item in allocations]}
    if phase == "pilot":
        path = out / "adapter.pt"
        temporary = out / "adapter.pending"
        trainer.torch.save([weight.detach().cpu() for weight in trainer.params], temporary)
        temporary.replace(path)
        report["weights_sha256"] = sha256(path)
    return report
