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

## Addendum 2026-11-06 (before the first P0 run; factory audit #1)

The decision rule, thresholds, model, hyperparameters, step counts and the
evaluation set are unchanged. These items only make the frozen setup runnable on
the factory node; none of them changes what is trained or how the gate is scored.

1. **In-training validation file.** Upstream solver scripts validate on
   `data/test_1200.parquet` (`trainer.test_freq=25`), which no upstream script
   creates (present only as a reference since upstream's initial commit).
   `prepare` now builds it with `python -m work3.val_subset`: 1200 rows of
   `data/test.parquet`, stratified by `data_source` with proportional
   largest-remainder allocation, numpy seed 20261106. It feeds training-time
   validation logs only; the gate still uses the full `test.parquet`.
2. **Offline data.** Upstream `process_train.py` / `process_test.py` always call
   `hf_hub_download`, and the factory mirror returns 404. With `TRAIN_PARQUET` /
   `TEST_PARQUET` set, `prepare` runs the unmodified scripts through
   `python -m work3.p0_launch offline-hf`, which returns the local
   `PeterJinGo/nq_hotpotqa_train` files instead of downloading. The factory
   verified the local `test.parquet` (51,713 rows) against the FlashRAG originals,
   row by row. Because upstream `process_test.py` only logs download errors,
   `prepare` now checks the output files explicitly.
3. **Evaluation batch size.** Evaluation sets `data.val_batch_size=256` (upstream
   default `null` = one batch of 51,713). Greedy decoding and EM are unchanged.
4. **Ports.** The retriever (upstream hard-codes 8000) and the proposer-reward
   SGLang server (8001) move to `RET_PORT` (default 8100) and `REWARD_PORT`
   (default 8101) because 8000 is used by another project on the factory node.
   The retriever runs unmodified through `python -m work3.p0_launch serve`, which
   only overrides the port passed to `uvicorn.run`. The tool config and
   `reward_kwargs.base_url` are rewritten accordingly. `patch` fails if any
   reference to 8000 / 8001 remains.
5. **Process cleanup.** Upstream `kill -9 $(lsof -t -i :8000)` lines become
   no-ops (`lsof` is absent and they would target other projects). Every process
   started by a stage carries `WORK3_P0_TAG=work3-p0` in its environment, and
   cleanup kills only tagged processes. The global `pkill` / `ray stop --force`
   calls are removed. `check` now fails if the two ports are taken or if other
   GPU compute processes are running (override: `ALLOW_BUSY_GPU=1`).
6. **Retriever readiness.** After each upstream retriever launch, `patch` inserts
   `scripts/p0/wait_retriever.sh`, which blocks until the retriever answers
   (max 30 min) or stops the stage. Upstream only `sleep 30`s, and the 61 GB
   index on NAS can take minutes to load. Searches sent before that would fail
   silently inside rollouts.

## Addendum 2026-11-09 (factory feedback #2, during iteration 1)

The decision rule, model, hyperparameters, step counts, data and evaluation set
are unchanged. These items make the setup runnable, or change memory use only.

1. **Retriever calls.** Upstream `retrieve_endpoint` always unpacks
   `(results, scores)`, so any request without `"return_scores": true` returns
   HTTP 500. verl's search tool always sends it; the readiness probe
   (`wait_retriever.sh`) now does too. Every manual call to the service must
   include it. The upstream file is not patched.
2. **Hydra config path.** Hydra resolves a relative `--config-path` against
   `verl/trainer/main_ppo.py`, not the working directory. `patch` now rewrites
   `CONFIG_PATH` in all nine launch scripts to `$DRZERO_DIR/config`, and
   evaluation uses the same absolute path. The config contents are unchanged.
3. **Environment pins.** These are the versions the factory verified:
   torch 2.6.0, sglang 0.4.6.post5, transformers 4.51.1,
   compressed-tensors 0.10.2, uvloop 0.21.0, numpy 1.26.4,
   faiss-gpu-cu12 1.8.0.2, flash-attn 2.7.4.post1 (ABI matching torch),
   pandas 2.3.3, tensordict 0.9.1.
   - compressed-tensors 0.15 imports a transformers module that does not
     exist in 4.51.
   - uvloop 0.23 breaks verl's `asyncio.get_event_loop()` in `fsdp_sglang.py`.
   - `check` now verifies these imports and behaviours.
4. **Memory fractions (throughput only).**
   - The proposer's in-trainer SGLang rollout uses
     `gpu_memory_utilization = CHAL_ROLLOUT_MEM`, default 0.45; upstream used
     0.25, shared with the reward server.
   - The proposer-reward SGLang server keeps 0.25.
   - Evaluation uses `EVAL_ROLLOUT_MEM`, default 0.6 (previously 0.5).

   These settings change KV-cache capacity and concurrency, not sampling. The
   factory's first proposer stage ran at 0.25; later stages use the new
   default. The stage timings record when the switch happened.
5. **Bookkeeping.** `timings.txt` lines now carry the exit code and end time.
   The retriever wrapper writes latency statistics to
   `logs/retriever_stats.json` (read-only observability). The read-only board
   (`scripts/run_board.sh`) displays both.

## Addendum 2026-11-09 (b) (factory feedback #3, iteration-1 proposer at step 16)

The decision rule, model, hyperparameters, step counts, data and evaluation are
unchanged. These items change only parallel layout and observability:

1. **In-trainer rollout tensor parallelism.** `ROLLOUT_TP`, default 1, was 2
   upstream. It applies to proposer generation, solver training rollouts and
   evaluation. On 4 GPUs this gives four full-model SGLang engines instead of
   two TP-2 engines. Sampling parameters and the model are unchanged; only
   floating-point reduction order differs, which is already non-deterministic.
   It takes effect from the iteration-1 solver stage onward. The
   iteration-1 proposer ran with TP 2.
2. **Reward-server layout made explicit.** The proposer-reward SGLang server
   used the launch scripts' shared `tp` / `dp` variables. It now uses
   `REWARD_TP` / `REWARD_DP`, which default to the upstream tp 2 × dp 2. It is
   also started with `--enable-metrics`, which only exposes Prometheus gauges.
   Changing the layout is allowed later if the speed probe shows a data-parallel
   group idling; it is a parallelism-only change.
3. **Observability.** Two read-only additions: `run_p0.sh probe`
   (`work3/speed_probe.py`) and a reward-server panel on the board. `patch` now
   writes the launch scripts atomically (temp file + rename), so re-patching
   while a stage runs cannot alter the script that stage is executing.
4. Batch sizes are deliberately not changed. Per the factory's timing, the
   update phase is about 5% of a proposer step, and GPU 0 has about 0.5 GB of
   headroom at the training peak.

## Addendum 2026-11-10 (factory feedback #4: OOM at iteration-1 proposer step 16)

The decision rule, model, hyperparameters, step counts, data and evaluation are
unchanged; this is an environment and resource incident.

1. **Incident.** Under p0-r2, the iteration-1 proposer crashed with CUDA OOM
   on GPU 0 at step 16/50 (2026-11-09 17:46). The crash happened in verl
   `fsdp_sglang` `wake_up` → FSDP `state_dict` (52 MiB free, 86 MiB needed).
   There was no checkpoint yet (upstream saves every 25 steps), so the run
   restarts from scratch. Rewards had risen from 0.128 to 0.356 by then.
2. **Cause.** The proposer's in-trainer SGLang memory fraction had been raised
   from the upstream 0.25 to 0.45 in addendum 2026-11-09 item 4. It bought no
   speed, because 94% of a proposer step is the separate reward server. It is
   re-occupied at every wake-up, and GPU 0 also hosts the retriever encoder
   and part of the reward server. The training-side peak grows as proposer
   responses lengthen.
3. **Decision.** `CHAL_ROLLOUT_MEM` returns to the upstream 0.25. Item 4 of
   addendum 2026-11-09 is withdrawn for the proposer; the evaluation fraction
   is unaffected.
4. **Crash recovery (bookkeeping only).**
   - Checkpoints every `SAVE_FREQ` = 10 steps, keeping the latest
     `CKPT_KEEP` = 2. The final step (50 / 100 / 150) is always saved.
   - Re-running `run_p0.sh iter<t>` skips sub-stages whose outputs exist.
   - Training resumes from the latest checkpoint (verl `resume_mode=auto`).
     For proposer iterations 2–3 this switch is made only when that
     iteration already has its own checkpoints; otherwise they start from the
     previous iteration's checkpoint as upstream does.
   - The training trajectory is unchanged.

## Addendum 2026-11-10 (b) (factory feedback #6: OOM at iteration-1 proposer step 30)

The decision rule, model, hyperparameters, step counts, data and evaluation are
unchanged.

1. **Incident.** Under p0-r7, the iteration-1 proposer hit CUDA OOM on GPU 0
   in the actor backward pass at step 30/50 (2026-11-10 13:21; 1.62 GiB
   requested, 1.47 GiB free). Memory had grown from 42 GB (step 1) to
   55–59 GB (step 28) as proposer responses lengthened; rewards rose from 0.128
   to 0.532. Checkpoints at steps 10 and 20 were intact; training resumes from
   `global_step_20`.
2. **Decision.** The per-GPU micro-batch for the actor update and log-prob
   passes goes from the upstream 2 to `TRAIN_MICRO` = 1, for proposer and
   solver. The mini-batch (256), number of optimiser steps, learning rate and
   all sampling are unchanged.
   - What changes: gradient accumulation goes from 32 to 64 micro-batches per
     GPU, so the per-pass token count, and with it the length-dependent memory
     peak, halves.
   - With `loss_agg_mode=token-mean`, the token mean is now taken per sample
     rather than per pair of samples. Upstream's pairing already differs from
     ours, because it depends on world size (8 GPUs vs 4); this changes the
     same detail.
3. **Not adopted.**
   - CPU retrieval (`FAISS_GPU=0`): a flat CPU index scans about 64 GB per
     query, and thousands of searches per step arrive in bursts.
   - `expandable_segments` stays an opt-in fallback (`EXPANDABLE_SEGMENTS=1`),
     because its interplay with SGLang's memory saver is unverified.
