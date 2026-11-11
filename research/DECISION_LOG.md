# Work 3 decision log

## 2026-11-05: Bootstrap after scoping

Scoping report: local `工作三_选题调研.md` (user's machine). Candidate A (PBRO) is
recommended, B (perturbation-invariant RL) is the fallback; C (training-free replay
instability) is at most an add-on; D/E are excluded. No protocol, gate, or run is
registered. Venue: ICML 2027 primary, SIGIR 2027 alternative, decide by 2026-12-15.

Decision: `BOOTSTRAP_REPOSITORY_SCOPING_ONLY`.

## 2026-11-05: Publish repository

The user chose a public GitHub repository from the start
(`https://github.com/CathyMuo/work3_byd`). Direction and venue still await
confirmation; nothing is registered.

Decision: `PUBLISH_PUBLIC_REPOSITORY`.

## 2026-11-05: Drop PBRO and re-scope

The user dropped direction A (PBRO) and asked for a new survey of hot, solid
directions, with recursive self-improvement (RSI) in view. The new local report is
`工作三_选题调研_v2_RSI.md`. The candidate area is zero-data self-evolving
(self-play) search agents, with Dr. Zero as the planned baseline. Method details
stay local until the user confirms. Nothing is registered.

Decision: `DROP_PBRO_RESCOPE_TO_SELF_EVOLVING_SEARCH`.

## 2026-11-06: Confirm direction; freeze P0 and P1

The user confirmed the re-scoped direction (zero-data self-evolving search agents; method details
private until arXiv; ICML 2027 primary). Dr. Zero is pinned at
`98c4a6dd2495e85d8b4d947ab4c030e540c922c5`.

- P0 frozen in `research/protocols/P0_drzero_reproduction.md`: PASS iff the
  iteration-3 seven-dataset average EM ≥ 0.306 (paper 0.326 − 0.020); computed only
  by `work3/p0_gate.py`.
- P1 frozen in a protocol file on local branch `private/p1-ncp` (unpublished until arXiv). SHA-256 of that file:
  `4727861905cb818eb5b39eb2874c22e350fc89a4ffb7751e5b33e57c4e7b6322`.

No P0 or P1 data exist at freezing time.

Decision: `FREEZE_P0_P1_GATES`.

## 2026-11-06: Rename repository and publish everything

At the user's request the GitHub repository was renamed from `work3_byd` to
`https://github.com/CathyMuo/work3` (old URLs redirect) and the home directory
moved to `/mnt/wh/RAG/work3`. The user chose to keep Work 3 fully public like
Work 2, including the P1 protocol and later method code. The private branch was
fast-forwarded into `main`; `research/protocols/P1_noise_diagnosis.md` is
unchanged, so its SHA-256 still equals the value recorded above (its header
sentence about living on a local branch is now historical).

Decision: `RENAME_TO_WORK3_AND_PUBLISH_ALL`.

## 2026-11-06: Fix the P0 workflow after factory audit #1 (tag `p0-r1`)

The factory audited the P0 workflow before any run. Verified on the pinned code
and fixed:

- Blocker: missing `data/test_1200.parquet`. `prepare` now builds a stratified 1200-row subset.
- Blocker: `prepare` failed offline. Local parquet files are now used through a runtime `hf_hub_download` substitute; upstream files are untouched.
- Risks: unbounded evaluation batch (now 256); global `pkill` that would kill another project's retriever (cleanup now kills only environment-tagged processes); hard-coded ports 8000/8001 (now configurable, default 8100/8101); missing `lsof` (upstream port kills disabled, ports checked in `check`).
- Added: the stage waits until the retriever answers, instead of upstream's fixed 30 s.

All of these are recorded in the P0 protocol addendum dated 2026-11-06. The gate
and training setup are unchanged.

Accepted for P1: the replay code must report per-rollout consistency between
assistant-turn token boundaries and the re-tokenised text (the rollout-alignment
caveat in the pinned upstream fix). This is reported only and is not a gate.

Decision: `P0_WORKFLOW_FIX_P0_R1`.

## 2026-11-06: Release the P1 implementation (tag `p1-r1`)

Added the P1 rollout and replay code (`work3/p1_agent.py`, `work3/p1_run.py`),
the frozen-rule summary (`work3/p1_summary.py`), the upstream prompt parity
check (`work3/p1_parity.py`), `scripts/p1/run_p1.sh` and
`docs/factory/P1_guide.md`. The summary script is committed before any P1
rollout, as the protocol requires.

Implementation details the protocol left open are recorded in its addendum
dated 2026-11-06. The text before the addendum still matches the recorded
SHA-256.

The factory audit's trajectory-consistency request is covered by
`p1_parity.py`. On synthetic episodes (multi-query search, 5-turn limit, no
tool call), the rendered prompts matched upstream incremental construction
4/4, in both text and token ids.

Decision: `RELEASE_P1_IMPLEMENTATION_P1_R1`.

## 2026-11-06: Freeze P2 and release its implementation (tag `p2-r1`)

P2 is frozen in `research/protocols/P2_ncp_pilot.md` before any P0, P1 or P2
data exist. Three arms differ only in the proposer difficulty term:

- Dr. Zero: reuses the P0 run.
- Beta smoothing: n = 7, a deliberately generous budget for the baseline.
- NCP: the Dr. Zero term × the signal share w estimated from two same-prefix
  replays per solver rollout, as in P1.

GO iff all three hold:

- iteration-3 average EM(NCP) − EM(Dr. Zero) ≥ 0.020;
- EM(NCP) − EM(Beta) ≥ 0.010;
- NCP shows no decline from iteration 2 to 3.

P2 runs only if P0 is PASS and P1 is GO.

Code:

- `work3/ncp_reward.py`: drop-in proposer reward.
- `work3/p2_gate.py`.
- `scripts/p2/run_p2.sh`: one git worktree per arm, sharing P0's seed, test and
  validation files.
- `run_p0.sh` gains `P0_REWARD_ARM`. The default behaviour is unchanged
  (checked by a patch dry run).
- `docs/factory/P2_guide.md`.

Decision: `FREEZE_P2_RELEASE_P2_R1`.

## 2026-11-09: Factory feedback #2 fixed (tags `p0-r2`, `p1-r2`, `p2-r2`)

P0 started at the factory on 2026-11-09. The factory found five deterministic
failures on the documented path; all were verified on the pinned code:

- the retriever returns 500 without `return_scores`, so the readiness probe
  never succeeded;
- Hydra resolves a relative `--config-path` against `main_ppo.py`;
- `sglang[all]` pulls compressed-tensors 0.15, which needs transformers ≥ 4.53;
- uvloop 0.23 breaks verl's `get_event_loop()`;
- a module-level typing alias in `p1_agent.py` fails on Python 3.9.

Fixes:

- probe sends `return_scores`;
- absolute `CONFIG_PATH`;
- pinned and checked environment;
- a `get_event_loop` fallback in `ncp_reward.py`;
- `typing` aliases (all modules import and all tests pass on 3.9).

The factory's two local edits are merged.

Throughput, in response to the user's request:

- The proposer's in-trainer SGLang memory goes from 0.25 to 0.45. The reward
  server stays at 0.25. Evaluation memory goes from 0.5 to 0.6.
- The retriever wrapper records latency, so time spent waiting on search can
  be measured.
- The low-power rollout tail is latency-bound (sequential 5-turn
  generate→search). It is documented as expected rather than "fixed".
- The `undefined function` warnings are proposer sampling errors, not a schema
  mismatch. They are left as is, because P0 must not change prompts.

Added a read-only progress board, as Work 2 has: `work3/board.py` and
`scripts/run_board.sh` (port 8766). It shows the research briefing from
`research/` and live progress from run directories. P0 protocol addendum
2026-11-09 records items that touch the setup. Gates are unchanged.

Decision: `FACTORY_FEEDBACK_2_FIXES_R2`.

## 2026-11-09: Board training curves (tags `p0-r3`, `p1-r3`, `p2-r3`)

The user asked for the board to show training curves and other metrics. Added a
"训练曲线" view to the read-only board:

- every numeric metric from verl's per-step console lines, parsed
  incrementally and cached, grouped and selectable;
- presets for common metrics, validation EM, and timing / throughput;
- three-iteration overlay, with the step axis reset within each iteration;
- smoothing and hover values;
- the proposer reward series;
- P2 reward-log curves: difficulty, w, η, k distribution, tokens and seconds;
- in-memory 10-second history of GPU power, utilisation and memory, and of
  retriever latency and request rate.

Run scripts are byte-identical to the r2 tags, so switching tags mid-run is
safe. No gate, protocol or experiment setting changed.

Decision: `BOARD_TRAINING_CURVES_R3`.

## 2026-11-09: Board redesign for clarity (tags `p0-r4`, `p1-r4`, `p2-r4`)

The user found the board too crude: P0 showed 22 flat stage chips, and the
research page only dumped project files. They asked for an intuitive progress
view and a research explanation clear enough to present to the advisor.

Progress tab:

- a research roadmap (P0 → P1 → P2 → main experiments → submission) with
  status, frozen gate and key result for each phase;
- one overall progress bar per run, with the remaining time and the expected
  finish clock time;
- a plain-language "now running" box showing the within-iteration step count;
- one lane per iteration with four segments (proposer training, question pool,
  solver training, evaluation), sized by duration and showing EM on the
  evaluation segment;
- raw stages, retriever, alerts and logs moved into a collapsible section.

New 研究说明 page served from `research/briefing.html`, a curated
advisor-level briefing:

- background, the problem and the evidence (Dr. Zero Table 3);
- the variance decomposition, the same-prefix replay and NCP, with two
  diagrams and an illustrative reward table;
- how it differs from related work;
- the staged design with gates;
- contributions, risks and timeline.

The previous file dump is now 项目记录.

Run scripts are unchanged since r2, so switching tags is safe while P0 runs.

Decision: `BOARD_REDESIGN_R4`.

## 2026-11-09: Speed analysis after factory feedback #3 (tags `p0-r5`, `p1-r5`, `p2-r5`)

Factory timing: a proposer step takes about 452 s, of which about 429 s (94%)
is the proposer-reward rollout (5 multi-turn solver attempts per question on
the separate reward SGLang server); the update takes about 23 s. Raising the
in-trainer rollout memory (r2) gave no measurable gain, because that is not
the engine doing the reward rollout. Batch sizes are not worth changing, and
GPU 0 is full at the training peak. HRPO's hop grouping works as designed: it
standardises rewards within `data_source` hop groups, but it is an advantage
computation, not a rollout scheduler.

The factory saw GPUs 2 and 3 idle. That contradicts SGLang 0.4.6.post5's DP
controller, which places DP1 on GPUs 2–3 and round-robins requests, so it is
not acted on blindly.

Changes:

- in-trainer rollout TP 2 → 1 (four engines; mainly speeds up solver stages
  and evaluation);
- reward server layout made explicit (`REWARD_TP` / `REWARD_DP`, upstream
  default kept) and started with `--enable-metrics`;
- read-only `run_p0.sh probe` (process-to-GPU map, reward-server layout and KV
  capacity, queueing, retriever latency, with a verdict);
- the board shows reward-server load and bases the remaining-time estimate on
  the latest five step times;
- atomic `patch`, so the factory can apply r5 without stopping the running
  stage.

The reward-server layout will be decided from the probe output: DP not used →
tp 1 × dp 4; KV-bound → revisit memory; latency-bound tail → accept.

Recorded in P0 addendum 2026-11-09 (b).

Decision: `SPEED_R5_TP1_AND_PROBE`.

## 2026-11-10: OOM at proposer step 16; restore 0.25 and add crash recovery (tags `p0-r6`, `p1-r6`, `p2-r6`)

The factory's p0-r2 run of the iteration-1 proposer hit CUDA OOM on GPU 0 at
step 16/50, inside verl's FSDP→SGLang weight hand-off. No checkpoint existed
(upstream saves every 25 steps). The cause is our own r2 change: raising the
proposer's in-trainer SGLang memory fraction to 0.45. It brought no speed-up,
since the time goes to the separate reward server, and it removed the
headroom GPU 0 needs as training-side activations grow with longer proposer
responses.

Changes:

- the fraction returns to the upstream 0.25;
- checkpoints every 10 steps, keeping the latest 2;
- re-running an iteration skips finished sub-stages and resumes training from
  the latest checkpoint; proposer iterations 2–3 switch to `resume_mode=auto`
  only when they already have their own checkpoints.

Verified with a patch dry run and a stub-script crash-recovery simulation.

The ConnectTimeout retries seen before the crash come from the reward client's
connection burst to the reward server. They are retried successfully and are
unrelated to the OOM. The KV cache is preallocated and does not grow with
request volume; the growing quantity is the training-side activation peak.

Recorded in P0 addendum 2026-11-10. Gate unchanged. The factory restarts P0
from iteration 1.

Decision: `OOM_FIX_R6`.

## 2026-11-10: Speed review after factory feedback #5 (tags `p0-r7`, `p1-r7`, `p2-r7`)

Factory results:

- p0-r6's TP-1 in-trainer rollouts cut a proposer step from 452 s to 274 s
  (39%), with even memory across the four GPUs. Kept.
- Probe: reward server running mean 9.7, queue 0, KV use 5%, every GPU busy
  26–30%. Data parallelism works and the server is latency-bound, so it stays
  at the upstream tp 2 × dp 2 (TP lowers single-sequence latency).

What remains is the synchronous-RL tail. A step waits for its slowest few
multi-turn trajectories: up to 5 generate→search turns with up to 2560 new
tokens each.

The factory's "batch by hop" proposal (A) and probe-then-batch (B) are
rejected. SGLang batches continuously, and concurrency is far from saturated,
so the step time ≈ the longest trajectory; running hop groups one after
another would add their tails. Raising `max_parallel_calls` (C) is irrelevant:
it limits tool calls per turn. HRPO's hop grouping acts only on the advantage
computation, as designed.

Fixes in r7:

- `check` ran `set_event_loop(None)` before `get_event_loop()`, which fails on
  every uvloop version. It now reproduces the runtime situation in a fresh
  interpreter.
- The retriever stats separate service time (requests arriving with ≤ 2
  others in flight) from end-to-end time including queueing; the probe judges
  retrieval only on the former.

Training path unchanged: patch output is identical to r6.

Decision: `SPEED_REVIEW_R7`.

## 2026-11-10: Second proposer OOM (step 30); micro-batch 1 and stage ETAs (tags `p0-r8`, `p1-r8`, `p2-r8`)

The iteration-1 proposer under p0-r7 hit OOM in the actor backward pass at
step 30. This was not the static SGLang reservation fixed in r6. It was the
training-side peak, which grew from 42 GB to 59 GB as proposer responses
lengthened (rewards 0.128 → 0.532). Checkpointing every 10 steps kept
`global_step_20`.

Fix: the actor-update and log-prob micro-batch goes 2 → 1. Same mini-batch and
optimiser steps; half the tokens per pass, so half the length-dependent peak.
The cost is negligible, since the update is about 5% of a step. The bound
holds for iterations 2–3 because sequence length is capped.

Rejected: CPU retrieval (a flat CPU index scans about 64 GB per query under
bursty load). `expandable_segments` is an opt-in fallback only.

The board, at the user's request, now shows:

- the current stage's remaining time and finish clock;
- per-segment and per-iteration totals (done and remaining);
- per-phase P0 / P1 / P2 remaining time, with P2 arms estimated from P0's
  measured stage times;
- a warning when any GPU exceeds 93% memory.

Recorded in P0 addendum 2026-11-10 (b).

Decision: `OOM2_MICROBATCH_R8`.

## 2026-11-10: Step ETA on the board's current-stage card (tags `p0-r9`, `p1-r9`, `p2-r9`)

The user saw only a percentage in the board's "正在进行" card. The r8 ETA line
was separate from the progress bar. It also needs the board backend restarted,
since the page reloads but the backend does not.

The remaining time and finish clock now sit directly under the step progress
bar, and the sampling bar shows its own remaining time. The source depends on
what is available:

1. the mean of the latest five step times;
2. otherwise the progress bar's own rate (tqdm), which also covers
   question-pool generation;
3. otherwise the stage's typical duration.

The page falls back to tqdm's text when an older backend lacks the field. The
guide now says to restart the board after updating.

Board only; run scripts identical to r8.

Decision: `BOARD_STEP_ETA_R9`.

## 2026-11-11: A nonzero exit after a complete final checkpoint is not a failure (tags `p0-r10`, `p1-r10`, `p2-r10`)

**What happened (factory feedback #7).**

- On p0-r8 the iteration-1 proposer finished 50/50 steps in 4 h 48 min. The
  micro-batch fix held: memory stayed flat at 59 GB, and the reward went from
  0.128 to 0.616.
- It saved a complete `global_step_50`.
- Three seconds later the TaskRunner raised "DataLoader worker ... is killed by
  signal: Killed" and the run exited 1, which stopped `run_p0.sh` before
  question-pool generation.
- The cgroup recorded no OOM kill.

**What the error is not.**

- It is not a validation failure. The proposer uses `test_freq=-1`, and
  "Final validation metrics: None" is printed unconditionally.
- The killed process is the train dataloader's prefetch worker. It is
  probably still alive after the loop because `StatefulDataLoader` keeps its
  iterator for checkpointing.
- The log does not show who sent the SIGKILL during shutdown. Ray reclaiming
  the actor's children is the likely candidate. It can recur at the end of any
  proposer or solver stage.

**Decision.**

- A training stage that exits nonzero is accepted when its final checkpoint is
  complete:
  - the tracker names the final step (it is written last);
  - all rank shards are present;
  - `data.pt` is present;
  - `huggingface/config.json` is present.
- Such an exit is logged to `teardown_warnings.txt` and recorded as done for the
  board.
- Skip-on-rerun uses the same check instead of "the actor directory exists".
- Cleanup now reports untagged SGLang / Ray / verl GPU processes. The factory
  had a leftover 48–53 GB SGLang process. Cleanup does not kill such processes.

**Not adopted.**

- Lowering `data.dataloader_num_workers`. The iteration-2 and iteration-3
  proposers resume from the previous iteration's `data.pt`, which was saved
  with 8 workers. Changing the worker count risks that state no longer loading,
  and it deviates from upstream for no gain.
- Changing Ray's memory monitor. Nothing points to it.

Training settings are identical to r8.

Decision: `TEARDOWN_EXIT_TOLERATED_R10`.

## 2026-11-11: Output checks for every P0 stage, and an unattended `all` command (tags `p0-r11`, `p1-r11`, `p2-r11`)

**What happened (factory feedback #8).** On p0-r10 the iteration-1
question-pool stage exited 137 (SIGKILL). Its pool was complete: 33,923 rows,
hop split 40/30/20/10. r10 had covered only the training stages, so the run
stopped again. The user requires that unattended overnight runs never stop on
such teardown noise.

**Decision.**

- **Shared output checks.** `work3/stage_check.py` checks the output of each
  stage kind:
  - **checkpoint:** the tracker, every rank shard, `data.pt` and the HF config;
  - **pool:** the parquet reads in full, has rows, and has no empty prompt,
    data_source or answer;
  - **pool check:** the result JSON has `ok` true;
  - **HF model:** the config and every indexed shard;
  - **eval dump:** it parses, and its per-source counts equal those of
    `data/test.parquet`.
- **`run_p0.sh checked`.** Every stage of an iteration and every evaluation
  runs through `checked`. A nonzero exit continues only when the check passes
  and the output was written after the stage started (`--since`). Such exits
  go to `teardown_warnings.txt`.
- **Skipping on a re-run.** Re-runs skip on the same checks instead of "the
  file exists". Evaluations are now skipped when complete.
- **Diagnostics.** A nonzero exit appends signal, cgroup memory and dmesg OOM
  lines to the stage log, to find the SIGKILL's source.
- **`run_p0.sh all`.** It runs iter1 eval1 iter2 eval2 iter3 eval3 eval0 gate
  and retries a failing stage once after cleanup (`MAX_ATTEMPTS`).
- **Board.** It marks such stages "⚠ 收尾报错", lists them, and ignores
  durations under 5% of a stage's default when estimating ETAs. The factory's
  pool line recorded only 7 s.

**Not adopted.** A pool check counts as done only when it passed, not "has a
result". Training a solver on a pool that failed its check would waste the
night and make P0 invalid.

**Open question.** The factory's timings line `iter1_gen_data 7 137
1794360721` says this attempt ran only 7 s and ended at 09:32 CST. The pool must
therefore come from an earlier attempt. The full timings file and log were
requested.

Training settings are identical to r8.

Decision: `STAGE_OUTPUT_CHECKS_R11`.
