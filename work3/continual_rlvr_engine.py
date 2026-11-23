"""Four-rank online-policy training and free generation for the frozen pilot."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from work3.continual_rlvr_data import score
from work3.continual_rlvr_protocol import BATCH_EVAL, CAP, GROUP, GUARD_FLOOR, LR, PREFIX


@dataclass(frozen=True)
class Rollout:
    prompt: tuple
    response: tuple
    reward: bool
    truncated: bool


class Engine:
    def __init__(self, model_path, splits, deadline):
        self.rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        if self.world != 4 or torch.cuda.device_count() != 4:
            raise RuntimeError("four visible CUDA devices required")
        self.device = torch.device("cuda", self.rank)
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16-capable CUDA devices required")
        dist.init_process_group("nccl")
        self.deadline = deadline
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16).to(self.device)
        self.eos = base.generation_config.eos_token_id
        self.eos_set = set(self.eos if isinstance(self.eos, list) else [self.eos])
        self.pad = self.tokenizer.pad_token_id
        if self.pad is None or None in self.eos_set:
            raise ValueError("tokenizer needs pad and EOS ids")
        self.prompts = {}
        for cases in splits.values():
            for case in cases:
                ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": case["prompt"]}], tokenize=True, add_generation_prompt=True)
                if not ids or len(ids) > 768:
                    raise ValueError(f"prompt exceeds the 768-token bound: {case['id']}")
                self.prompts[case["id"]] = tuple(ids)
        torch.manual_seed(23071)
        self.model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.ddp = DistributedDataParallel(self.model, device_ids=[self.rank], broadcast_buffers=False)
        self.params = [param for param in self.model.parameters() if param.requires_grad]
        self.initial = [param.detach().clone() for param in self.params]
        self.reset()

    def check(self):
        if monotonic() >= self.deadline:
            raise TimeoutError("frozen wall-time cap reached")

    @torch.no_grad()
    def reset(self):
        for param, initial in zip(self.params, self.initial):
            param.copy_(initial)
        self.new_optimizer()

    def new_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.params, lr=LR, weight_decay=0.0)

    def gather(self, value):
        result = [None] * self.world
        dist.all_gather_object(result, value)
        return result

    def left_pad(self, prompts):
        width = max(map(len, prompts))
        ids = [([self.pad] * (width - len(prompt))) + list(prompt) for prompt in prompts]
        mask = [([0] * (width - len(prompt))) + ([1] * len(prompt)) for prompt in prompts]
        return torch.tensor(ids, device=self.device), torch.tensor(mask, device=self.device)

    @torch.no_grad()
    def generate(self, cases, seed, prefixes=None):
        self.check()
        self.model.eval()
        torch.manual_seed(seed)
        task = cases[0]["task"]
        plain = [self.prompts[case["id"]] for case in cases]
        prefixes = prefixes or [()] * len(cases)
        if len({len(prefix) for prefix in prefixes}) != 1 or any(case["task"] != task for case in cases):
            raise ValueError("a generation batch must have a single task and prefix length")
        ids, mask = self.left_pad([prompt + prefix for prompt, prefix in zip(plain, prefixes)])
        cap = CAP[task] - len(prefixes[0])
        output = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, use_cache=True, max_new_tokens=cap,
            eos_token_id=self.eos, pad_token_id=self.pad,
        ))[:, ids.shape[1]:].cpu().tolist()
        rollouts = []
        for case, prompt, prefix, new_tokens in zip(cases, plain, prefixes, output):
            stop = next((i + 1 for i, token in enumerate(new_tokens) if token in self.eos_set), None)
            truncated = stop is None
            response = prefix + tuple(new_tokens[:stop])
            text = self.tokenizer.decode(response, skip_special_tokens=True)
            rollouts.append(Rollout(prompt, response, not truncated and score(case, text), truncated))
        return rollouts

    def evaluate(self, cases, samples, seed, report=None):
        assigned = cases[self.rank::self.world]
        selected = [case for case in assigned for _ in range(samples)]
        rows = {case["id"]: [] for case in assigned}
        tokens, truncations = 0, 0
        for offset in range(0, len(selected), BATCH_EVAL):
            batch = selected[offset:offset + BATCH_EVAL]
            rollouts = self.generate(batch, seed + 131 * self.rank + 1021 * offset)
            for case, row in zip(batch, rollouts):
                rows[case["id"]].append(row.reward)
                tokens += len(row.prompt) + len(row.response)
                truncations += row.truncated
            if report and (offset % (8 * BATCH_EVAL) == 0 or offset + len(batch) == len(selected)):
                report(offset + len(batch), len(selected))
        parts = self.gather((rows, tokens, truncations))
        return ({key: value for part, _, _ in parts for key, value in part.items()},
                sum(part[1] for part in parts), sum(part[2] for part in parts))

    def log_probs(self, rows, *, train):
        prompts, masks = self.left_pad([row.prompt for row in rows])
        width = max(len(row.response) for row in rows)
        targets = torch.tensor([list(row.response) + [self.pad] * (width - len(row.response)) for row in rows],
                               device=self.device)
        active = torch.arange(width, device=self.device)[None, :] < torch.tensor(
            [len(row.response) for row in rows], device=self.device)[:, None]
        ids = torch.cat((prompts, targets), dim=1)
        mask = torch.cat((masks, active), dim=1)
        # Match the absolute positions used by generation for left-padded prompts.
        positions = mask.long().cumsum(dim=1) - 1
        positions.masked_fill_(mask == 0, 0)
        model = self.ddp if train else self.model
        logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                       use_cache=False, logits_to_keep=width + 1).logits[:, :-1]
        logprobs = -F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
        return logprobs * active, int(ids.numel())

    @torch.no_grad()
    def prefix_log_probs(self, rows):
        short = [Rollout(row.prompt, row.response[:PREFIX], False, False) for row in rows]
        return self.log_probs(short, train=False)[0].sum(dim=1).tolist()

    def policy_loss(self, rows):
        logp, cost = self.log_probs(rows, train=True)
        rewards = torch.tensor([row.reward for row in rows], dtype=torch.float32, device=self.device)
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-4)
        lengths = torch.tensor([len(row.response) for row in rows], device=self.device)
        return -(advantages * logp.sum(dim=1) / lengths).mean(), cost

    def bank_loss(self, packs, arm):
        rows = [entry[0] for pack in packs for entry in pack]
        if not rows:
            raise ValueError("empty anchor update")
        logp, cost = self.log_probs(rows, train=True)
        if arm == "trace_replay":
            lengths = torch.tensor([len(row.response) for row in rows], device=self.device)
            return -(logp.sum(dim=1) / lengths).mean(), cost
        current = logp[:, :PREFIX].sum(dim=1)
        losses = []
        start = 0
        for pack in packs:
            stop = start + len(pack)
            weights = torch.tensor([entry[1] if arm == "guard" else 1.0 for entry in pack],
                                   device=self.device).log()
            base_weights = torch.tensor([entry[3] if arm == "guard" else 1.0 for entry in pack],
                                        device=self.device).log()
            baseline = torch.tensor([entry[2] for entry in pack], device=self.device)
            log_mass = torch.logsumexp(current[start:stop] + weights, dim=0)
            floor = torch.logsumexp(baseline + base_weights, dim=0) + torch.log(
                torch.tensor(GUARD_FLOOR, device=self.device))
            losses.append(F.relu(floor - log_mass) / PREFIX)
            start = stop
        return torch.stack(losses).mean(), cost

    def update(self, primary, auxiliary, arm=None):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.ddp.no_sync():
            loss, primary_cost = self.policy_loss(primary)
            (0.5 * loss).backward()
        if arm in {"trace_replay", "uniform_prefix", "guard"}:
            loss, auxiliary_cost = self.bank_loss(auxiliary, arm)
        else:
            loss, auxiliary_cost = self.policy_loss(auxiliary)
        (0.5 * loss).backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(self.params, 1.0)):
            raise FloatingPointError("nonfinite gradient")
        self.optimizer.step()
        return primary_cost + auxiliary_cost

    def save(self, path):
        if self.rank == 0:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)


def fixed_order(cases, seed):
    result = list(cases)
    random.Random(seed).shuffle(result)
    return result


def rollout_tokens(rows):
    return sum(len(row.prompt) + len(row.response) for row in rows)
