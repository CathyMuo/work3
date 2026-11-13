# P2 protocol: noise-calibrated proposer reward, 3B pilot (FROZEN)

Frozen: 2026-11-06, before any P0, P1 or P2 data exist. Do not edit after the
first P2 run starts; corrections go in a new dated section at the end.

## Preconditions

P2 runs only if P0 is `PASS` and P1 is `GO`. Otherwise this protocol lapses.

## Question

Does replacing Dr. Zero's success-rate proposer reward with a noise-calibrated
version (NCP) improve the self-evolved solver, beyond a compute-generous
smoothing baseline?

## Arms

All arms are identical to P0 in every respect except the proposer's difficulty
term. Identical means: Qwen2.5-3B-Instruct, pinned Dr. Zero `98c4a6d`, the same
proposer seed file, test and validation files (symlinked from P0), hop ratio
4:3:2:1, HRPO proposer and GRPO solver, 50 + 50 steps per iteration, 3
iterations, and the same evaluation. The final proposer reward stays
`0.5 × format + difficulty`. Implementation: `work3/ncp_reward.py`.

| Arm | Solver rollouts per question | Difficulty term |
| --- | --- | --- |
| Dr. Zero | 5 (upstream engine) | 𝕀(0<k<n)·(n−k)/(n−1). **The P0 run is this arm**; it is not rerun. |
| Beta | **7** | 𝕀(0<k<n)·(1−p̃)/(1−p̃_min), with p̃ = (k+1)/(n+2) and p̃_min = 2/(n+2) |
| NCP | 5, plus 2 replays per rollout | Dr. Zero term × w, with w = max(0, σ²_tot − σ²_noise)/σ²_tot. σ²_tot = k(n−k)/(n(n−1)). σ²_noise = (1/n)Σᵢ(R̃ᵢ₁−R̃ᵢ₂)²/2 from replays built exactly as in P1 (same prefix, passages permuted per query block with a non-identity permutation, final assistant turn regenerated); unreplayable rollouts repeat their own outcome. |

Beta's 2 extra full multi-turn rollouts are meant to be at least comparable in
cost to NCP's 10 final-turn regenerations. The actual completion tokens of both
arms are reported from the reward logs. If NCP's proposer-reward completion
tokens exceed 1.5× Beta's, the result is reported as not compute-matched. The Beta and NCP arms
use `work3/p1_agent.py` for solver rollouts; its prompts were checked identical
to upstream construction (P1 addendum, item 2), and P1's parity report covers
the factory run.

One run (seed) per arm; the P0 run serves as the single Dr. Zero run.

## Gate (computed only by `python -m work3.p2_gate`)

Let A_t(arm) be the iteration-t seven-dataset average EM (same definition as P0).

| Item | Rule |
| --- | --- |
| Validity | P0 verdict is PASS; no arm summary is INVALID; iteration-2 and iteration-3 averages exist for all arms. Otherwise `INVALID`. |
| **Primary (all required)** | (a) A_3(NCP) − A_3(Dr. Zero) ≥ **0.020**; (b) A_3(NCP) − A_3(Beta) ≥ **0.010**; (c) A_3(NCP) ≥ A_2(NCP) (no decline). → `GO`, else `NO-GO`. |
| Secondary (reported, does not change GO/NO-GO) | P1-style η over 0<k<n questions of each arm's pool 3 (`run_p2.sh p1 <arm>`): η(NCP) ≤ η(Dr. Zero) − 0.10. Per-dataset iteration-3 EM. Reward compute (requests, completion tokens, seconds) for Beta and NCP. |

## Decision

`GO` → register the main experiments (more seeds, 7B, ablations) in a new
protocol before running them. `NO-GO` → stop NCP as the Work 3 method and
re-scope with the user. Thresholds are not revisited after results.

## Addendum 2026-11-13: exploratory search-behaviour report (written before any P2 data)

P0 showed that iterations 1 to 3 lose multi-hop EM because the solver
retrieves the gold answer less often (factory feedback #11, #12). The P2
summary therefore also reports `work3/p0_diag.py` on every arm's evaluation
dumps, per arm and iteration, for single- and multi-hop questions:

- search calls and queries per question;
- the share of questions whose gold answer appears in a retrieved passage;
- the split of wrong answers into "gold never retrieved" and "gold retrieved".

This report is exploratory. It is not part of the gate, and the arms, the
setup, the GO rule and the verdict are unchanged.
