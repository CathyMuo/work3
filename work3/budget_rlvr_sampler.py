"""Frozen Qwen sampling with both continuations forked from one KV cache."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

from work3.budget_rlvr_metrics import LONG, PREFIX, SHORT
from work3.continual_rlvr_data import last_box, load_prompt_tokenizer, prompt_ids, score

PROMPT_SHORT = (
    "<|im_end|>\n<|im_start|>user\n"
    "Use the reasoning already given. Give your best current exact rational answer now, "
    "in \\boxed{}, without further reasoning.<|im_end|>\n<|im_start|>assistant\n"
)


@dataclass
class Decode:
    ids: list
    ended: list
    cache: object
    mask: torch.Tensor
    last: torch.Tensor


class Sampler:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.pad = tokenizer.pad_token_id
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        if self.pad is None or None in self.eos:
            raise ValueError("tokenizer needs padding and EOS tokens")
        self.instruction = tokenizer.encode(PROMPT_SHORT, add_special_tokens=False)
        if not self.instruction:
            raise ValueError("short-branch instruction did not tokenize")

    @classmethod
    def load(cls, model_path, rank):
        if torch.cuda.device_count() != 4:
            raise RuntimeError("the frozen pilot requires four visible GPUs")
        torch.cuda.set_device(rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the frozen pilot requires BF16-capable GPUs")
        device = torch.device("cuda", rank)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16)
        return cls(model.to(device), load_prompt_tokenizer(model_path), device)

    def pad_prompts(self, prompts):
        width = max(map(len, prompts))
        ids = [[self.pad] * (width - len(prompt)) + list(prompt) for prompt in prompts]
        mask = [[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts]
        return torch.tensor(ids, device=self.device), torch.tensor(mask, device=self.device)

    @torch.inference_mode()
    def decode(self, ids, mask, cache, cap, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        ended = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
        generated = []
        for _ in range(cap):
            positions = (mask.cumsum(1) - 1).clamp_min(0)[:, -ids.shape[1]:]
            outputs = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                 past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = outputs.past_key_values
            probs = outputs.logits[:, -1, :].float().softmax(dim=-1)
            new = torch.multinomial(probs, 1, generator=generator)
            ids = torch.where(ended[:, None], self.pad, new)
            generated.append(ids)
            cache_mask = mask
            mask = torch.cat((mask, (~ended).long()[:, None]), dim=1)
            for eos in self.eos:
                ended |= ids[:, 0] == eos
            if bool(ended.all()):
                break
        tokens, stops = [], []
        for row in torch.cat(generated, dim=1).tolist():
            stop = next((n + 1 for n, token in enumerate(row) if token in self.eos), None)
            tokens.append(row[:stop])
            stops.append(stop is not None)
        # The last sampled token has not been forwarded into the cache yet.
        return Decode(tokens, stops, cache, cache_mask, ids)

    @torch.inference_mode()
    def branches(self, jobs, seed):
        prompts = [prompt_ids(self.tokenizer, case) for case, _ in jobs]
        ids, mask = self.pad_prompts(prompts)
        prefix = self.decode(ids, mask, None, PREFIX, seed)
        rows = []
        for index, ((case, repeat), tokens) in enumerate(zip(jobs, prefix.ids)):
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
            right = score(case, raw)
            ended = prefix.ended[index]
            rows.append({"qid": case["id"], "repeat": repeat, "prompt_tokens": len(prompts[index]),
                         "prefix_ids": tokens, "prefix_ended": ended, "prefix_correct": right,
                         "prefix_has_box": bool(last_box(raw)), "base_tokens": len(prompts[index]) + len(tokens),
                         "short_ids": [], "long_ids": [], "short_ended": ended, "long_ended": ended,
                         "short_correct": ended and right, "long_correct": ended and right,
                         "short_tokens": 0, "long_tokens": 0})
        unfinished = [index for index, ended in enumerate(prefix.ended) if not ended]
        if not unfinished:
            return rows
        indices = torch.tensor(unfinished, device=self.device)
        cache = prefix.cache
        cache.batch_select_indices(indices)
        mask = prefix.mask.index_select(0, indices)
        last = prefix.last.index_select(0, indices)
        extra = torch.tensor(self.instruction, device=self.device)[None, :].expand(len(unfinished), -1)
        short_ids = torch.cat((last, extra), dim=1)
        short_mask = torch.cat((mask, torch.ones_like(short_ids)), dim=1)
        short = self.decode(short_ids, short_mask, deepcopy(cache), SHORT, seed + 1073741824)
        long_mask = torch.cat((mask, torch.ones_like(last)), dim=1)
        long = self.decode(last, long_mask, cache, LONG - PREFIX, seed + 2147483648)
        for slot, index in enumerate(unfinished):
            row, case = rows[index], jobs[index][0]
            short_answer = self.tokenizer.decode(short.ids[slot], skip_special_tokens=True)
            long_answer = self.tokenizer.decode(row["prefix_ids"] + long.ids[slot], skip_special_tokens=True)
            row.update(short_ids=short.ids[slot], long_ids=long.ids[slot], short_ended=short.ended[slot],
                       long_ended=long.ended[slot], short_correct=score(case, short_answer),
                       long_correct=long.ended[slot] and score(case, long_answer),
                       short_tokens=len(self.instruction) + len(short.ids[slot]), long_tokens=len(long.ids[slot]))
        return rows
