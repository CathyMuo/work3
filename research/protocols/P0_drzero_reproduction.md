# P0 protocol: Dr. Zero Qwen2.5-3B reproduction (FROZEN)

Frozen: 2026-11-06, before any P0 data exist. Do not edit after the first P0 run
starts; corrections go in a new dated section at the end, and the original gate
stays as the decision rule.

## Purpose

Establish that the factory can reproduce the published Dr. Zero self-evolution
loop (arXiv 2601.07055 v2, Table 3) on 4×H800, and produce the artifacts the P1
diagnosis needs. P0 is a reproduction check, not a result of this project.

## Fixed setup

- Upstream: `facebookresearch/drzero` at commit
  `98c4a6dd2495e85d8b4d947ab4c030e540c922c5` (2026-03-23, "fix rollout inconsistency").
  Code is used unmodified except for the launch-script substitutions made by
  `scripts/p0/run_p0.sh` (GPU count, data-parallel size, conda activation, local
  model and retriever paths, optional CPU FAISS, an optional lower memory
  fraction for data generation if it runs out of memory, and
  `trainer.total_training_steps`).
- Model: Qwen2.5-3B-Instruct. Hop ratio 4:3:2:1. HRPO proposer (group 1, reward
  rollouts 5), GRPO solver (group 5). 50 proposer steps and 50 solver steps per
  iteration (paper Tables 5 and 6). The upstream scripts train one epoch over a
  much larger file, so the step count is set explicitly with
  `trainer.total_training_steps` = 50 / 100 / 150 for proposer iterations 1 / 2 / 3
  (the proposer resumes across iterations) and 50 for every solver iteration.
- Retrieval: wiki-18 + e5-base-v2 Flat index, top-3, as in the upstream scripts.
- Hardware: one node, 4×H800. `train_batch_size` 256 and all other
  hyperparameters as upstream.
- Seed: upstream defaults (no extra seed control); one run.

## Evaluation

- Solver checkpoints `solver_iter{1,2,3}_hf`, plus the base model as iteration 0
  (reported only).
- Test file: the merged Search-R1 test set produced by upstream `process_test.py`
  (seven datasets: NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue,
  Bamboogle), full sets, greedy decoding as upstream validation, exact match from
  the dumped per-sample `score`.
- Average = unweighted mean of the seven per-dataset EM values (this reproduces
  the paper's "Average" column: Dr. Zero 3B iteration 3 = 0.326).
- Computed only by `python -m work3.p0_gate` from the validation dumps.

## Gate

| Item | Rule |
| --- | --- |
| Validity | All seven datasets present in each evaluated dump; every score is 0 or 1; iteration-3 dump present. The solver training pools must contain solver prompts (no proposer seed document); checked with `python -m work3.pool_check`. Any failure → `INVALID` (fix and rerun; not a FAIL). |
| **Primary** | Iteration-3 average EM ≥ 0.326 − 0.020 = **0.306** → `PASS`, else `FAIL`. |
| Recorded, not gating | Iteration 1 / 2 averages (paper 0.304 / 0.319), per-dataset deltas, dataset counts vs the FlashRAG sizes, wall-clock time per stage. |

On `FAIL`: check environment versions and the substitutions first; if a clean
rerun still fails, report to the home server. Switching codebase (e.g. SSP) is a
new decision, not part of this protocol.

## Artifacts to keep (needed by P1)

- Solver training pools `data/zero_challenger_iter{1,2,3}_*.parquet`.
- Solver HF checkpoints `solver_iter{1,2,3}_hf` (and the base model path).
- Proposer checkpoint `global_step_150` (FSDP) — keep until P2 is decided.
- Validation dumps and stage logs (proposer reward lines included).

Only `p0_summary.json`, the pool check output and stage timings are returned to
the home server.
