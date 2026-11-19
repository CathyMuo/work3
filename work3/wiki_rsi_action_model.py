"""Small paired-action update with the same LoRA configuration for both trained arms."""

import math
import random

import torch
from torch import nn
from torch.nn import functional as F

RANK = 8
SCALE = 2.0
SEED = 20261119
EPOCHS = 2
ACCUMULATION = 4
LEARNING_RATE = 5e-5
PAIR_WEIGHT = 0.5


class ProjectionWithAdapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(RANK, base.in_features, device=base.weight.device))
        self.up = nn.Parameter(torch.zeros(base.out_features, RANK, device=base.weight.device))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            projected = F.linear(x, self.down.to(x.dtype))
            result = result + SCALE * F.linear(projected, self.up.to(x.dtype))
        return result


def attach_adapters(model):
    model.requires_grad_(False)
    adapters = []
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            attention = layer.self_attn
            projection = getattr(attention, name)
            if not isinstance(projection, nn.Linear):
                raise TypeError("Expected Qwen2 query and value linear projections")
            adapter = ProjectionWithAdapter(projection)
            setattr(attention, name, adapter)
            adapters.append(adapter)
    if not adapters:
        raise ValueError("No decoder projections found")
    return adapters


def trainable(adapters):
    return [weight for adapter in adapters for weight in (adapter.down, adapter.up)]


def scheduled_indices(n):
    rng = random.Random(SEED)
    order = []
    for _ in range(EPOCHS):
        order.extend(rng.sample(range(n), n))
    return [order[start:start + ACCUMULATION] for start in range(0, len(order), ACCUMULATION)]


def mean_logprobs(model, prompt, completions, pad, device):
    sequences = [prompt + completion for completion in completions]
    width = max(map(len, sequences))
    ids = torch.tensor([seq + [pad] * (width - len(seq)) for seq in sequences], device=device)
    attention = torch.arange(width, device=device) < torch.tensor([len(seq) for seq in sequences], device=device)[:, None]
    labels = ids.masked_fill(~attention, -100)
    labels[:, :len(prompt)] = -100
    to_score = labels[:, 1:] != -100
    hidden = model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    selected = hidden[:, :-1][to_score]
    logits = model.get_output_embeddings()(selected)
    losses = F.cross_entropy(logits.float(), labels[:, 1:][to_score], reduction="none")
    return torch.stack([-part.mean() for part in losses.split([len(item) for item in completions])])


def objectives(mean_logp, frozen_margin):
    positive, negative = mean_logp
    sft = -positive
    preference = F.softplus(-(positive - negative - frozen_margin))
    return sft, sft + PAIR_WEIGHT * preference


def train_arm(model, adapters, rows, margins, pad, device, *, paired, progress):
    model.train()
    params = trainable(adapters)
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0)
    schedule = scheduled_indices(len(rows))
    losses = []
    for step, group in enumerate(schedule, 1):
        total = torch.zeros((), device=device)
        for index in group:
            row = rows[index]
            likelihoods = mean_logprobs(model, row["prompt"], (row["positive"], row["negative"]), pad, device)
            positive_loss, paired_loss = objectives(likelihoods, margins[index])
            # The unused loss keeps the same pair-shaped computation in the positive control.
            loss = paired_loss if paired else positive_loss + paired_loss * 0.0
            (loss / len(group)).backward()
            total += loss.detach()
        if not torch.isfinite(total).item():
            raise RuntimeError("Non-finite independent pilot loss")
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(round(total.item() / len(group), 6))
        progress(step, len(schedule))
    consumed = EPOCHS * sum(2 * len(row["prompt"]) + len(row["positive"]) + len(row["negative"]) for row in rows)
    return {"steps": len(schedule), "pair_presentations": EPOCHS * len(rows), "unpadded_input_tokens": consumed,
            "first_loss": losses[0], "last_loss": losses[-1]}


@torch.no_grad()
def generate_queries(model, tok, inputs, device, max_new):
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = model.generation_config.eos_token_id or tok.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    width = max(map(len, inputs))
    padding = [width - len(ids) for ids in inputs]
    tokens = torch.tensor([[pad] * n + ids for n, ids in zip(padding, inputs, strict=True)], device=device)
    mask = torch.arange(width, device=device) >= torch.tensor(padding, device=device)[:, None]
    result = model.generate(input_ids=tokens, attention_mask=mask, do_sample=False, use_cache=True,
                            max_new_tokens=max_new, eos_token_id=sorted(eos_ids), pad_token_id=pad)
    decoded = []
    for row in result[:, width:].tolist():
        ended = next((index for index, token in enumerate(row) if token in eos_ids), None)
        continuation = row if ended is None else row[:ended]
        decoded.append({"text": tok.decode(continuation, skip_special_tokens=True),
                        "finish": "length" if ended is None else "stop", "tokens": len(continuation)})
    return decoded
