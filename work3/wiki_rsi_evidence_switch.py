"""Build evidence-conditioned query preferences from two verified retrieval branches."""

import torch
import torch.nn.functional as F

from work3 import wiki_rsi_action_model as action_model
from work3 import wiki_rsi_natural_query as natural_query
from work3 import wiki_rsi_replay as replay
from work3 import wiki_rsi_w2 as wiki
from work3.evidence_ref import text_key, title_key


def verified_example(question, shared, branches):
    """Use exactly two observed branches. Return None when the query preference does not flip."""
    first, second = branches
    if not question.strip():
        raise ValueError("Expected a question for both evidence states")
    queries = [branch["query"].strip() for branch in (first, second)]
    facts = [(title_key(name), text_key(sentence)) for name, sentence in
             (first["witness"], second["witness"])]
    if (len({text_key(query) for query in queries} - {""}) != 2
            or any(not all(fact) for fact in facts) or len(set(facts)) != 2):
        raise ValueError("Expected two distinct queries and two distinct evidence witnesses")
    states = [[*shared, *branch["observation"]] for branch in (first, second)]
    returns = [branch["returned"] for branch in (first, second)]

    def coverage(passages):
        return {fact for fact in facts if wiki.measure(passages, [fact])["all_missing_facts"]}

    before = [coverage(state) for state in states]
    if before != [{facts[0]}, {facts[1]}]:
        return None
    gains = [[len(coverage([*state, *docs]) - already) for docs in returns]
             for state, already in zip(states, before, strict=True)]
    if gains != [[0, 1], [1, 0]]:
        return None
    return {"question": question, "states": states, "queries": queries, "gains": gains}


def encode_example(tokenizer, example):
    """Render exactly the visible Wiki search state, keeping training witnesses out of the prompt."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Missing end-of-sequence token")
    prompts = [natural_query.prompt_ids(tokenizer, example["question"], visible)
               for visible in example["states"]]
    completions = [tokenizer(query, add_special_tokens=False)["input_ids"] + [eos]
                   for query in example["queries"]]
    if any(not 1 < len(tokens) <= natural_query.MAX_NEW for tokens in completions):
        return None
    if any(not prompt or len(prompt) + max(map(len, completions)) > replay.solver.MAX_MODEL_LEN
           for prompt in prompts):
        return None
    return {"prompts": prompts, "completions": completions}


def score_example(model, example, pad, device):
    return torch.stack([action_model.mean_logprobs(model, prompt, example["completions"], pad, device)
                        for prompt in example["prompts"]])


def losses(scores):
    """Return independent preference and evidence-switch objectives on the same four scores."""
    if scores.ndim < 2 or scores.shape[-2:] != (2, 2):
        raise ValueError("Expected scores for two evidence states and two actions")
    margins = torch.stack((scores[..., 0, 1] - scores[..., 0, 0],
                           scores[..., 1, 0] - scores[..., 1, 1]), dim=-1)
    per_state = F.softplus(-margins).mean()
    # Additive prompt-only and query-only biases cancel in the difference of the two margins.
    interaction = F.softplus(-margins.mean(dim=-1)).mean()
    return per_state, per_state + interaction
