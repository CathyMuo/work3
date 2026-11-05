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
