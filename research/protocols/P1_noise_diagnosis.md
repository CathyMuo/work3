# P1 protocol: behaviour-independent noise in self-play proposer rewards (FROZEN)

Frozen: 2026-11-06, before any P0 or P1 data exist. This file lives on a local
branch only; its SHA-256 is recorded in the public `research/DECISION_LOG.md` so
that the freeze can be verified after publication. Do not edit after the first
P1 rollout; corrections go in a new dated section at the end.

## Hypothesis

H1: in Dr. Zero self-play, a substantial part of the solver-outcome variance on
proposer questions does not depend on the solver's search behaviour, and this
share is larger on the questions the proposer reward favours most (k = 1 of 5).

## Inputs (from P0)

- Pools: `zero_challenger_iter1_*.parquet` (pool 1) and
  `zero_challenger_iter3_*.parquet` (pool 3).
- Reward solver of each pool, i.e. the solver the proposer was trained against:
  base Qwen2.5-3B-Instruct for pool 1, `solver_iter2_hf` for pool 3.
- Same retriever, prompts, tool config and EM checker (`em_check` against the
  single proposer answer) as Dr. Zero's proposer reward.

## Sampling

- From each pool, draw 500 questions uniformly at random (numpy seed 20261106),
  among rows with a non-empty question and answer.
- If, after the rollouts below, either analysis group (k = 1, or k ∈ {3,4})
  has fewer than 60 questions in a pool, draw 500 more with seed 20261107, then
  20261108 (at most 1500 per pool). Stop as soon as both groups reach 60.

## Rollouts

For every sampled question q and its reward solver:

1. Originals: n = 5 independent solver rollouts, temperature 1.0, top-p 1.0,
   max 5 assistant turns, as in the proposer reward rollout. R_i ∈ {0,1} by EM.
   k = Σ R_i.
2. Behaviour b_i = every assistant turn before the final answering turn,
   with its tool responses (queries and the retrieved top-3 passages).
3. Replays: for each original rollout, m = 2 replays. Each replay keeps b_i's
   tokens, applies an independent uniformly random **non-identity** permutation
   to the three passages inside every tool response (passage text unchanged),
   and re-generates from the start of the final assistant turn at temperature
   1.0. R̃_i1, R̃_i2 ∈ {0,1} by the same EM. If a replay issues a new search
   call, it continues normally within the remaining turn budget; the rate is
   recorded.
4. Rollouts with no tool call keep an empty b_i; their replays only re-generate
   the answer. Truncated rollouts score 0 and are replayed from the same point.

## Estimates

- σ²_tot(q) = k (n − k) / (n (n − 1)) (unbiased, from originals).
- σ²_noise(q) = (1/n) Σ_i (R̃_i1 − R̃_i2)² / 2 (unbiased within-behaviour
  variance, from the replay pair only, so it does not reuse R_i).
- For a group G of questions with 0 < k < n:
  η(G) = Σ_{q∈G} σ²_noise(q) / Σ_{q∈G} σ²_tot(q).
- Uncertainty: question-level bootstrap, 2000 resamples, numpy seed 20261106,
  percentile 95% intervals. Pools 1 and 3 are pooled for the primary test
  (resampling within each pool).
- Lucky-hit check: for each k = 1 question, 16 further original rollouts;
  p̂16 = successes / 16.

## Gate

| Item | Rule |
| --- | --- |
| Validity | Replays issuing a new search call ≤ 20% of all replays; group sizes reached as above; pools pass the P0 pool check. Otherwise `INVALID` (redesign, not NO-GO). |
| **Primary (both required)** | Pooled over pools 1 and 3: (a) η(k = 1) ≥ **0.35**; (b) η(k = 1) − η(k ∈ {3,4}) ≥ **0.10**, with the bootstrap 95% interval of the difference above 0. → `GO`, else `NO-GO`. |
| Secondary (reported, does not change GO/NO-GO) | (c) η over all 0 < k < n questions of pool 3 ≥ that of pool 1 (Goodhart signature); (d) share of k = 1 questions with p̂16 ≤ 1/16 ≥ 30%; replay-only-order vs answer-resampling breakdown is out of scope for P1. |

Interpretation recorded in advance: under a pure behaviour-independent model η is
about 0.75–0.8 in both groups; under fully behaviour-determined outcomes η = 0.
Criterion (b) therefore requires the reward-maximal questions to be noisier than
mid-pass-rate ones, not merely that noise exists.

## Decision

`GO` → implement the noise-calibrated proposer reward and run P2 (gate frozen
separately before P2 data). `NO-GO` → stop this direction and re-scope with the
user. Results are summarised by a script committed before the P1 rollouts start;
the factory returns only that summary.

## Addendum 2026-11-06 (before any P1 data; implementation `p1-r1`)

The hypothesis, sampling, rollout counts, estimates, thresholds and decision rule
above are unchanged. These items fix details the protocol left open. All were
written before any P0 or P1 data existed.

1. **Passage permutation with several queries.** Each search call may contain
   several queries, so a tool response holds three passages per query. The
   non-identity permutation is drawn independently for each query's top-3
   block. Passage text is unchanged and the "Doc i" labels follow the new order.
   Seeds: `numpy.random.default_rng([20261106, pool, qid, i, j])`.
2. **Prompt construction.** `work3/p1_agent.py` re-renders the full conversation
   with the chat template at every turn. Tool-call parsing mirrors SGLang
   0.4.6.post5 `Qwen25Detector` plus verl's argument decoding; tool responses
   mirror verl `search_r1_like_utils`; sampling and length limits mirror Dr.
   Zero's proposer reward rollout. A home-server check found the prompts
   identical (text and token ids) to upstream `AsyncRolloutRequest`'s incremental
   construction on synthetic episodes. `work3/p1_parity.py` repeats the check on
   the factory rollouts and reports the result (reported only, not a gate).
3. **Unreplayable originals.** Some originals end without a generated final
   assistant turn, because the context filled after a tool response or before
   generation. For these, both replays take the original outcome, so they add
   zero noise; this is a conservative choice. The count is reported.
4. **Failed searches.** Tool responses without passages (search error or empty
   query list) are kept unchanged in replays and counted.
5. **Rates.** The new-search rate is computed over all replays, including
   unreplayable ones. The group-size rule applies to each pool separately.
6. **Serving.** Each pool's reward solver is served by SGLang with tp 1 and
   dp = number of GPUs; this affects memory only. Both pools use the
   Qwen2.5-3B-Instruct tokenizer and chat template, which the solver checkpoints
   inherit.

The text of this file before this addendum has the SHA-256 recorded on
2026-11-06 (`4727861905cb818eb5b39eb2874c22e350fc89a4ffb7751e5b33e57c4e7b6322`).
