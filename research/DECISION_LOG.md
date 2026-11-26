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

## 2026-11-11: Root cause of the stalled P0 run: background servers held the stage's log pipe (tags `p0-r12`, `p1-r12`, `p2-r12`)

**Evidence.** Factory feedback #9 sent `timings.txt`, the question-pool log
and the pool's mtime. They show:

- **Proposer.** It finished at 11-10 19:54 with exit code 0. The stage was
  recorded only at 11-11 09:31:54, 66,706 s after it started. Feedback #7's
  "exit 1" was a misreading of a Ray teardown traceback. The "leftover
  48–53 GB SGLang" was the proposer-reward server, still running.
- **Question pool.** It was written at 14:18 after a 4 h 32 min generation,
  and recorded at 15:28:33.
- **Two short-lived next stages.** Each time, the next stage died 7 s
  (generation, 09:32) or 4 s (solver, 15:28) after the stage before it ended,
  with exit 137. That is the SIGKILL pass of a manual `run_p0.sh cleanup`
  started 10 s earlier.

**Cause.** The upstream scripts launch their retriever and reward server with
`&` and never stop them. `timed` ran `cmd | tee log`. The background servers
inherited the pipe, so `tee`, and with it the stage, waited until they died.
The factory's cleanup killed the servers. That released the stalled
orchestrator, which started its next stage just in time for the same cleanup's
SIGKILL.

**Decision.**

- **No pipe.** Stages write straight to their log file, and the console
  follows it with `tail -f`. A regression test leaves a background process
  running and requires the stage to return immediately. r11's code waits for
  that process.
- **`cleanup` stops orchestrators first.** It SIGKILLs other `run_p0.sh`
  stage runs for the same `DRZERO_DIR` before killing tagged processes.
- **One orchestrator per output directory.** A `flock` lock enforces it.
  Children of `all` share the parent's lock, and stage processes do not
  inherit the lock fd.
- **Board.** Recorded durations above 1.5× a stage's default are replaced by
  the log's evidence: 50 × mean step time, or the finished progress bar's
  elapsed time, plus 10 minutes. This is for the ETA estimates.

**Earlier releases.** The r10 and r11 output checks stay as a safety net.
They were not the cause.

Training settings are identical to r8.

Decision: `NO_PIPE_STAGE_LOGS_R12`.

## 2026-11-13: P0 PASS; multi-hop decline is the open question; start P1 (tag `p1-r13`)

**Result (factory feedback #10).**

- **Gate:** iteration-3 7-dataset average EM 0.3071 ≥ 0.306 → **PASS**. The
  gate is frozen and the verdict stands.
- **Average EM, iterations 0/1/2/3:** 0.1284 / 0.3168 / 0.3103 / 0.3071.
  Paper: – / 0.304 / 0.319 / 0.326.
- **Execution:** the whole run was unattended on r12. It took 35 h 39 min:
  27 stages, 4 automatic retries, no manual intervention.

**Reading.**

1. **Only multi-hop departs from the paper.** Single-hop mean
   (nq, triviaqa, popqa) went 0.426 → 0.455, +0.029 (paper +0.034).
   Multi-hop mean (hotpotqa, 2wiki, musique, bamboogle) went 0.235 → 0.196,
   −0.039 (paper +0.013). Our iteration-1 multi-hop score is above the
   paper's (0.235 vs 0.207); our iteration-3 score is below it (0.196 vs 0.220).
2. **The average decline is small and noisy.** It is −0.0097 from iteration 1
   to 3. Bamboogle (125 questions, −0.056) carries most of it. The average of
   the other six datasets is almost flat: 0.332 → 0.330. The
   evaluation-sampling SE of the 7-dataset average is about 0.0055, mostly
   from bamboogle. The per-dataset shifts on the large sets are well beyond
   that noise: hotpotqa −0.027 on 7,405 questions, 2wiki −0.038 on 12,576,
   popqa +0.054 on 14,267.
3. **This is one seed.** The paper also reports one run. Run-to-run variance
   is unknown, so the trend difference is not yet attributable.
4. **Hop distributions say nothing about question quality.** The 4:3:2:1 hop
   split of each pool is fixed by the seed file (`search_zero_<n>` prompts),
   so it carries no information about the questions the proposer wrote.

**Candidate causes (factory A–D).**

- **C (only 50 incremental steps) and D (solver continues from the previous
  solver)** are upstream's design. Both are reproduced exactly, so neither
  alone explains a difference from the paper.
- **A (noisy proposer reward)** is what P1 measures. P1 groups questions by k
  (successes out of n), not by hop.
- **B (multi-hop questions drift out of the solver's reach)** shows up in P1's
  k histograms of pool 1 versus pool 3.
- **E (new): the solver searches less.** Multi-hop questions need several
  searches. If self-play training rewards answering with fewer searches,
  multi-hop accuracy falls while single-hop rises. The P0 eval dumps already
  contain every trajectory, so this can be checked on CPU now.

**Decision.**

- **Start P1 now.** Its gate is unchanged.
- **Before P1, run `work3/p0_diag.py`** on the P0 eval dumps. It reports, per
  dataset and iteration: searches per question, no-search share, no-answer
  share, EM by search count, and paired iteration 1→3 flips with exact
  McNemar tests. It tests E.
- **P1 summary adds an exploratory hop breakdown** (P1 Addendum 2026-11-13,
  written before P1 data). It addresses whether A and B differ by hop. It is
  not a gate.

**What follows from P1.**

- **GO:** P2 proceeds as frozen. P0's run is the Dr. Zero arm:
  A1/A2/A3 = 0.3168/0.3103/0.3071. NCP must reach A3 ≥ 0.3271 and A3 ≥ A2.
  A GO alone does not show that noise causes P0's multi-hop decline. The hop
  breakdown and p0_diag decide how far the paper can claim that.
- **NO-GO:** the NCP direction stops, as frozen. The next direction is then
  chosen from the diagnosis, for example if E holds. That is a new survey and
  decision, made then, not now.

Decision: `P0_PASS_START_P1_R13`.

## 2026-11-13 (b): Search decline confirmed, cause open; check our own micro-batch deviation first (tags `p0-r14`, `p1-r14`)

**Evidence (factory feedback #11, `p0_diag` v1).**

- **Search calls per question fell at every iteration.**
  - Single-hop: 1.20 → 1.14 → 1.08.
  - Multi-hop: 1.56 → about 1.48 → 1.40.
  - Paired, iteration 1→3: −0.12 and −0.15 per question.
- **Multi-hop questions lost ground overall.** From iteration 1 to 3, 2,609
  flipped correct → wrong and 1,840 wrong → correct (exact McNemar
  p = 8e-31).
- **Multi-hop EM by search count.** EM with 2 calls rose from 0.315 to 0.363.
  EM with 1 call fell from 0.300 to 0.239, while the share of 1-call
  trajectories grew from 62% to 74%.
- **Statistics leave out repeated question text.** Questions whose text
  repeats within a dataset (about 2,100, mostly popqa) are dropped, so the
  per-dataset EMs differ slightly from the gate's.

**What the factory's reading over-states.**

1. **Not "noise vs. strategy".** E is a solver behaviour. A (a noisy
   proposer reward) is a possible upstream driver of what the solver is
   trained on. The two are not exclusive, and P1 still tests A as frozen.
2. **Search *calls* are not retrieval.** One call may carry several queries,
   and v1 did not count queries.
3. **EM by search count is observational.** The model chooses when to search
   again, so these buckets are selected, not assigned. The 1-call bucket's
   falling EM fits a shift of answerable questions to premature answers, but
   does not prove it.
4. **"chars" is the solver's response length on benchmark questions**, not
   question length. It says nothing about B.
5. **Transcription error.** The factory's table contained values (`0.000`)
   that the script's 2-decimal format cannot print. The JSON file is now
   requested instead.

**Our own deviation, a candidate cause.** verl's default `loss_agg_mode`
`token-mean` averages over the tokens of one micro-batch, then over
micro-batches.

- p0-r8 cut the micro-batch from 2 to 1 to avoid an OOM, for proposer steps
  21–150 and every solver. Each response is then length-normalised on its
  own.
- This per-sequence weighting is the bias Dr. GRPO describes. Short correct
  responses gain per-token weight and long wrong ones lose penalty. That
  pushes toward fewer searches.
- Upstream's micro-batch 2 normalises over random pairs, which is weaker.
- The 2026-11-10 note called this difference "tiny". Given E, that judgement
  must be tested, not assumed.

**Decision.**

1. **`p0_diag` v2 (CPU).** It reports queries per question and per call, and
   whether the gold answer (from `data/test.parquet`) appears in any retrieved
   passage. It splits wrong answers into "gold never retrieved" (premature
   stop, or search went astray) and "gold retrieved" (reading or reasoning
   failure), and reports flips by search change.
2. **P1 runs as frozen.** It is unchanged and runs next.
3. **`run_p0.sh ablate` after P1.**
   - It retrains the iteration-3 solver exactly as in P0 (from
     `solver_iter2_hf` on pool 3) twice: once with micro-batch 2, once again
     with micro-batch 1 as a noise floor.
   - It evaluates both and compares them with P0's iteration-3 solver.
   - Outputs go to separate directories.

**Pre-registered rule for the ablation, written before any ablation data.**

- **Runs:**
  - M1 = P0's iteration-3 solver;
  - R1 = the micro-batch-1 retrain;
  - R2 = the micro-batch-2 retrain.
- **Metrics:**
  - EM_mh = the mean over hotpotqa, 2wiki, musique and bamboogle of
    per-dataset EM, from `p0_diag_ablate.json`;
  - S_mh = multi-hop search calls per question.
- **Noise floor:** d0 = |EM_mh(R1) − EM_mh(M1)|.
- **CONTRIBUTES** if EM_mh(R2) − EM_mh(R1) ≥ max(0.015, 2·d0) and
  S_mh(R2) > S_mh(R1). Then:
  - our micro-batch deviation contributes to the multi-hop decline;
  - P2 cannot reuse P0 as the Dr. Zero arm under the current setting;
  - how to restore upstream's weighting without the OOM is decided then.
- **NO EVIDENCE** otherwise. Then E is treated as a property of the Dr. Zero
  self-play loop in our setup.
- **Limits.** One retrained iteration measures the solver-side effect of one
  iteration only. It cannot cover the cumulative effect over three
  iterations, or the effect on the proposer.

**What does not change.**

- The P0 verdict.
- The P1 protocol and gate.
- The P2 protocol.

P2 does not start before both P1 and the ablation are in.

Decision: `DIAG_V2_AND_MICROBATCH_ABLATION_R14`.

## 2026-11-13 (c): Board shows progress, ETA and curves for every stage (tags `p0-r15`, `p1-r15`, `p2-r15`)

**Problem.** The user reported that training curves appeared only for the
iteration-1 proposer. Other stages showed no curves and no usable progress.

**Gaps found.**

- **Evaluation and P1 had no progress signal.** Evaluation stages and P1
  pools fell back to "typical duration" guesses.
- **The step-line parser was fragile.** It failed on numpy-2 / tensor reprs
  (`np.float64(0.5)`), and the ETA path ignored them.
- **Ablation stages were not on the board.**

**Changes.** No experiment setting changes; this is the board, plus one
start-time file per stage written by `run_p0.sh` and `run_p1.sh`.

- **Progress per stage kind:**
  - training: steps / 50, ETA from the latest five step times;
  - question-pool generation: the generation bar, ignoring model-loading and
    CUDA-graph bars;
  - evaluation: verl's per-batch "validation generation end" lines, 203
    batches of 256 questions;
  - P1 pools: question records and lucky re-runs, from the JSONL files, with
    the k histogram so far;
  - other stages: typical duration since `logs/<stage>.start`.
- **ETA fallback.** When no start time exists, the board estimates the rate
  from its own observations.
- **Curves.** Every training stage is listed with its recorded step count.
  The numeric parser accepts numpy / tensor reprs, and the page flags step
  lines it cannot parse.
- **New curve sections.** Evaluation EM curves per dataset and iteration,
  with the paper's averages. An ablation table compares P0's iteration-3
  solver with the retrained micro-batch 2 and 1 solvers.
- **Ablation card.** A separate "P0 消融" run card has its own lanes and
  compare overlay.
- **Verification.** Checked on a synthetic run directory in the factory's log
  formats (plain, numpy-2, and ANSI-coloured step lines; generation and
  evaluation logs; a running ablation; a running P1 pool) in the browser.
  There are 8 new unit tests.

Decision: `BOARD_ALL_STAGES_R15`.

## 2026-11-13 (d): Multi-hop loss is a retrieval-recall loss; order unchanged (P1, then the ablation)

**Evidence (factory feedback #12, `p0_diag` v2).** The JSON was announced but
not attached; the numbers below are the factory's.

| Multi-hop, iteration | EM | gold retrieved (R) | wrong, never retrieved | wrong, retrieved |
| --- | --- | --- | --- | --- |
| 1 | 0.281 | 0.470 | 0.481 | 0.238 |
| 3 | 0.246 | 0.404 | 0.547 | 0.206 |

**Decomposition.**

- Accuracy given retrieval, (R − wrong & retrieved) / R, is unchanged:
  0.494 at iteration 1, 0.490 at iteration 3.
- Accuracy without retrieval, (1 − R) − wrong & never retrieved, is 0.049 at
  both.
- So ΔEM ≈ 0.49 × ΔR = −0.032, against an observed −0.035. The multi-hop loss
  is almost entirely lower recall, not worse reading.
- Single-hop recall rose from 0.692 to 0.712, and accuracy given retrieval
  from 0.643 to 0.670.
- Queries per multi-hop question fell about 21%, from 1.556 × 1.153 = 1.79 to
  1.404 × 1.009 = 1.42.
- Gold retrieval is a lenient proxy: the normalised gold string appears in a
  retrieved passage.

**What the factory's reading gets wrong.**

1. **Queries per call collapsed mostly from the base model to iteration 1**
   (1.45–1.82 → about 1.0). That is the iteration whose EM rose 2.5×, and
   whose no-answer share fell from about 0.50 to 0.02–0.09. The base model
   mostly overran the 2,560-token response budget. Fewer queries per call is
   largely the model learning to fit that budget, which is the upstream
   setting the paper used too. It cannot by itself explain iteration 1→3.
   From iteration 1 to 3, multi-hop queries per call moved only 1.15 → 1.01;
   fewer calls and fewer queries together cut recall.
2. **Tool responses are not truncated.** At the pinned commit, upstream's
   SGLang rollout path, the one Dr. Zero uses, does not reference
   `max_tool_response_length` (`sglang_rollout.py` and `schemas.py` were
   checked), and the search tool's formatting has no length limit.
   Truncation therefore does not explain the query collapse.
3. **"chars" is the solver's response length.** It is still not evidence for B.

**Decision.**

1. **P1 first, then `run_p0.sh ablate`, as decided in (b).**
   - P1 is the frozen, half-day test of whether NCP's premise holds.
   - The ablation decides whether P0's multi-hop decline comes from our
     micro-batch deviation.
   - P2 needs both, so the shorter decision goes first.
   - The GPUs are currently used by Work 2.
2. **NCP does not pivot to fix a length bias.**
   - Loss normalisation (Dr. GRPO) is prior work, not our contribution.
   - If the ablation finds CONTRIBUTES, the corrected weighting becomes the
     shared base of all P2 arms. The Dr. Zero arm is then re-run under it, and
     NCP is tested on top only if P1 is GO.
   - If NO EVIDENCE, recall collapse is a property of the self-play loop. It is
     then a finding in its own right, worth a direction of its own if P1 is
     NO-GO.
3. **P2 reports the search behaviour of every arm** (P2 Addendum
   2026-11-13, exploratory). Whether NCP changes recall is then measured
   rather than argued.
4. **P0 in the paper.** Report it as a reproduction within the frozen
   tolerance (0.307 vs 0.326). Disclose in the appendix:
   - the full per-iteration, per-dataset trajectory;
   - the multi-hop divergence from the paper and the recall decomposition;
   - our micro-batch deviation, with the ablation result.

Decision: `RECALL_DECOMPOSITION_ORDER_UNCHANGED`.

## 2026-11-16: P1 NO-GO; the NCP direction stops; run the ablation, then re-survey (tags `p0-r16`, `p1-r16`, `p2-r16`)

**Result (factory feedback #14; `research/results/P1_summary_20261116.json`).**

| | pool 1 (base solver) | pool 3 (solver_iter2) | pooled |
| --- | --- | --- | --- |
| questions | 1,500 | 1,000 | 2,500 |
| eta(k=1) | 0.680 | 0.679 | 0.680 |
| eta(k in {3,4}) | 0.692 | 0.742 | 0.718 |
| difference | −0.012 | −0.063 | −0.039, 95% CI [−0.123, +0.049] |

- **Primary (a)** holds: eta(k=1) ≥ 0.35.
- **Primary (b)** fails. Its CI upper bound, +0.049, is far below the
  required +0.10, and both pools are negative.
- **Secondary:** (c) false; (d) 0.293 < 0.30.
- **Validity:** parity 6,000/6,000; new-search replays 18.7% < 20%; no
  problems.
- **Verdict: NO-GO.** Per the frozen rule, the NCP direction stops and P2 is
  cancelled.

**Reading.**

1. **The noise share is high but flat in k.** It is about 0.7 in every k
   group and in every hop group (the hop CIs overlap). NCP's weight
   w = 1 − eta would therefore be nearly constant, and NCP would reduce to a
   rescaled Dr. Zero reward. The mechanism NCP needs, noise concentrated
   where the reward is highest, is absent. This is more than a missed
   threshold.
2. **eta ≈ 0.7 should not be quoted as an intrinsic limit of self-play.**
   - It measures the variance from regenerating the final turn at
     temperature 1 with permuted passages.
   - Pool 1's own new-search rate, 28.3%, inflates it.
   - Unreplayable originals, counted as zero noise, deflate it.
3. **The pool comparison is confounded.** Pool 1 and pool 3 differ in both
   the pool and the reward solver.
4. **Few questions carry a training signal.** Even solver_iter2 never solves
   55% of pool-3 questions and always solves 21.6%. Only about 23% (k 1–4)
   give a nonzero GRPO advantage; pool 1 with the base solver gives 35%.
5. **P1 does not explain P0's multi-hop recall collapse.** It rejects one
   specific mechanism, the proposer drawn to lucky k = 1 questions.

**Decision.**

1. **Stop NCP; cancel P2.** The P2 guide is marked cancelled and the board's
   roadmap shows it.
2. **Run `run_p0.sh ablate` next.** Its pre-registered rule is in 2026-11-13
   (b). It is now the key input for the next direction:
   - **CONTRIBUTES:** the recall collapse is largely our micro-batch
     artifact. It is not a direction by itself.
   - **NO EVIDENCE:** "retrieval-recall collapse in self-evolving search
     agents" enters the re-survey as the lead candidate.
3. **Re-survey for a new direction, as frozen.** Inputs:
   - the P0 recall decomposition;
   - P1's flat noise;
   - the low share of informative pool questions;
   - the ablation verdict.

   The direction is chosen after the ablation. Nothing else runs at the
   factory until then.
4. **Records are kept.** Whether P0 and P1 appear in a new paper, for example
   as a negative result on reward noise, is decided with the new direction.
5. **Venue.** ICML 2027 (about 01-16 abstract) is now tight. The 12-15 venue
   decision stands; SIGIR 2027 remains the alternative.

Decision: `P1_NO_GO_STOP_NCP`.


## 2026-11-16 (b): Micro-batch ablation NO EVIDENCE; diagnose multi-hop evidence acquisition

Factory feedback #15 supplied the complete JSON regenerated from the two evaluation
dumps. Archive: [P0 ablation](results/P0_ablation_20261116.json).

| Run | Multi-hop EM (four-dataset macro) | Multi-hop EM (pooled) | Gold-string hit (pooled) | Search calls/question (pooled) |
| --- | ---: | ---: | ---: | ---: |
| P0 iter1 | 0.235033 | 0.280549 | 0.469904 | 1.556079 |
| M1: P0 iter3 | 0.196096 | 0.246391 | 0.404025 | 1.403678 |
| R1: micro-batch 1 | 0.190959 | 0.243060 | 0.394962 | 1.372141 |
| R2: micro-batch 2 | 0.194571 | 0.248612 | 0.396695 | 1.212411 |

The preregistered macro EM gap is +0.003612, below
max(0.015, 2 × 0.005138) = 0.015. Pooled search calls fall by 0.159730
instead of rising. Both requirements fail. **NO EVIDENCE.** The returned
table's +0.005552 uses pooled EM. The reported McNemar tests compare each
retrain with M1 and do not establish equivalence. The gate covers the third
solver iteration from one shared checkpoint and pool, leaving earlier and
proposer-side effects unresolved.

Both retrains have around 39.5–39.7% gold-string hits versus 47.0% for
iteration 1. Search-call count alone cannot describe retrieval behavior.
The ablation does not identify the effect of stopping, query content, or
which training data caused the decrease. Check bridge and answer evidence
on benchmarks with annotated supporting documents.

Decision: **Make evidence acquisition across self-play rounds the lead
candidate.** Start with paired, offline step-level diagnosis on existing
traces. If warranted, replay both checkpoints on common states with the
same retriever, first observation, and query/token budget. Separate the
decision to continue from the evidence acquired by a query. To test a
self-play-specific claim, cross early/late checkpoints with early/late
question pools or use a matched fixed-pool control. Set confirmatory metrics
and gates before collecting intervention data. Question-pool signal sparsity
remains a fallback candidate. No new training is scheduled.

[SearchMaster](https://arxiv.org/html/2608.01822) already studies self-play
depth drift. [Retrieval, Reward, and Training Protocols](https://arxiv.org/html/2605.27881v1)
already reports query-recall decreases and uses matched search history in
supervised search RL. A Work3 method needs evidence of a self-play feedback
mechanism and a repair that improves evidence coverage and EM at matched cost.

Automation: six train/convert/eval stages returned zero, followed by an
empty text report and absent JSON. The exact report-process failure is
unknown. The launcher now writes stdout and stderr to a named log and
offers a CPU-only summary command to recover from existing dumps.

Decision: P0_ABLATION_NO_EVIDENCE_DIAGNOSE_EVIDENCE_ACQUISITION.


## 2026-11-17: Register the offline retrieval diagnosis after factory feedback #16

The factory reproduced the ablation summary from saved dumps without using GPUs.
The registered ablation verdict is NO EVIDENCE. The seven-dataset 0.310700
reported today is the deduplicated diagnostic EM; the formal full evaluation
remains 0.3071. Archived 2->micro2 paired multi-hop counts are 1,874 losses and
1,661 gains, McNemar p=0.000362; the hand-transcribed line differs. The independent
micro1 retrain also drops from iteration 2. Neither test identifies the effect
of earlier training rounds or establishes equivalence of the micro-batches.

Run the new CPU-only retrieval diagnostic on the saved P0 trajectories. The
primary endpoint is paired change in coverage of all annotated support titles
at query budget 2, macro-averaged over HotpotQA and 2Wiki. Fixed budgets cut
observed prefixes; neither forced continuation nor a new training run is
involved. The protocol records coverage quality, exact sentence corroboration,
and cautious bridge/answer roles. The first-query miss cohort may prioritize
which replay to build; it is descriptive until controlled.

The factory retains full results and sends the compact summary. Home owns any
follow-up script and launch decision. [Factory guide](../docs/factory/REPLY_20261117_fb16.md)
and [frozen protocol](protocols/P0_retrieval_diagnosis.md).

Protocol SHA-256: 13be447d4afcd7ce355ba3d63a539025d88e04758e9a0b9efcc26a77dfc9a80d


## 2026-11-17 恢复固定 Wikipedia 索引主线

库存合成试验归入历史协议诊断。停止安排订单引用试验。主问题是能否用真实检索可修复的失败，改进下一轮自博弈任务分配及求解。先查同历史补证据的作用，再验证原索引的受预算查询是否可达，最后在独立训练与评测页面上比较同预算的真反馈和打乱反馈。

厂内首轮只执行 W0 CPU 输入审计。原跨轮 P0 取证比较的 INVALID_INPUT 判定保留。此轮重新建立单轮真实轨迹候选框，不从可解析子集推断 P0 退化。脚本、冻结输入和回传格式见 Wikipedia RSI 厂内指南。

## 2026-11-18 家服冻结 W1 回放

将研究目标收束为同训练预算下，哪些真实搜索失败带来跨页面学习收益。W0 与 W1 先核验有效输入、解码历史和证据特异救回。新增真实无关段落与空列表对照，独立记录裸历史原答案复现。W0 每数据集须有 60 个候选，W1 每数据集至少 40 个可回放题，门槛在首次真实运行前确定。W1 只用公开支持句做诊断。

等 W1 读数后决定是否验证原索引的实际查询，之后再安排等预算训练和跨页面迁移。[家服指南](../docs/RSI_wikipedia_home.md) · [冻结协议](protocols/RSI_wikipedia.md)

## 2026-11-18 反馈 22 后复审 HotpotQA 引用包

初次 W0 未通过。HotpotQA 的 7,175 条有效轨迹里，5,138 条触发支持句不齐，另有 2 条未匹配引用。2,035 条完整；其中 1,282 道做错，63 道已有完整支持句，1,209 道在同轮返回里找不齐缺句的来源，10 道符合候选条件。2Wiki 总池 393 道，60 是抽样上限。

家服逐题核对了 7,405 条 HotpotQA 官方 fullwiki 和 distractor dev 记录。旧包 5,317 道缺支持句，补齐官方原句后剩两道。W0b 在相同评测输入、缓存和门槛下验证这个输入修复。如候选仍不足，再评估扩展真实缓存来源。W1 完整回放、后续真实搜索和训练待各自准入。库存合成指针实验继续停用。[厂内 W0b 指南](../docs/factory/RSI_wikipedia_W0b.md)

## 2026-11-18 反馈 23 后执行 W0c

W0b 的 HotpotQA 完整引用由 2,035 增至 7,172，同轮合格候选 10 增至 43。4,938 道缺证据失败中仍有 4,895 道找不齐缓存来源。2Wiki 整套分母复现。保留原双数据集门槛，执行一次 0–2 轮真实缓存扩展，并验证同轮基线和来源散列。两侧达标才执行 CPU 历史重建。若缓存扩展仍不足，后续以原索引上的前瞻性受预算检索诊断为候选，避免继续用旧日志的可恢复性替代实际动作验证。[厂内 W0c 指南](../docs/factory/RSI_wikipedia_W0c.md)

## 2026-11-18 反馈 24 后启动 W1 单卡回放

W0c 同轮对照 43/393 均复现，扩展池增至 288/1,137。原始准备汇总给出缺证据配对 56/59 道、证据齐全 30/29 道，全部 174 道。五道缺证据题排除，另有一道证据齐全题排除。跨轮缓存扩大可干预样本，W1 估计的是给定缓存来源与可重建历史下的直接补证据效应。

复用原准备文件和 P0 第 3 轮 3B 检查点，先做 CPU 启动检查和单卡四题冒烟，再跑全部 578 次生成。冒烟完成门槛独立于答题正确率。依据完整配对结果及格式敏感性和来源分层安排 W2 的实际查询。[厂内 W1 指南](../docs/factory/RSI_wikipedia_W1.md)

## 2026-11-18 反馈 25 后启动 W2a

W1 完成 578 次生成。真证据对等长无关段落的配对收益为 HotpotQA +26.8 个百分点，2Wiki +33.9 个百分点。三臂格式均有效时分别为 +24.4 和 +28.6 个百分点。两边各有 12 道证据特异救回。该结果建立了可审计状态下直接补证据的作用。较早缓存与同轮来源分层方向相同，HotpotQA 同轮只有 11 道，无法从分层读数判断来源效应大小。

先用原索引测一条真实查询能否取到缺句，区分原题、历史查询与支持页标题诊断。待取得证据后再做冻结作答，不扩大同类补证据回放。[厂内 W2a 指南](../docs/factory/RSI_wikipedia_W2a.md)

## 2026-11-18 反馈 26 后区分输出接口与查询内容

原检索锚点 8/8 复现。标题诊断命中任一缺句 104/115，取齐全部 93/115。严格历史查询仅执行 25/115，全部缺句命中 3/115。一次查询是当前诊断预算。固定这一预算，比较同字节解析及原生接口，并分层统计可见标题线索。先读取真实输出和命中，推断查询失败的成因；随后决定线索因果干预或冻结作答。[厂内诊断指南](../docs/factory/RSI_wikipedia_W2_query_contract.md)

同日增加可选多卡推理，把原生提示批次分给最多三个空闲 H800 上的独立副本，旧提示留在原卡。包括 CPU 输入审计、逐条索引校验、生成和检索的耗时分别测量。维持既定样本、查询预算和判据。

## 2026-11-18 反馈 27 后分解生成截断与目标线索

原严格与同输出兼容的取齐都是 3/115，原生提示为 5/115。原生臂有 52/115 达到 64 token 长度上限，33 条执行中仍有 28 条未取齐。下一轮同题测 64 对 192，同时在 192 诊断目标标题与匹配的其他标题。看板改为独立实验运行页、全局轮询、动作与进程心跳，早期 W0 从原目录只读呈现。[厂内指南](../docs/factory/RSI_wikipedia_W2_budget.md)

## 2026-11-18 反馈 28 后验证真实返回的作答收益

同提示 192 消除了本轮 49 次截断，但完整取证从 6/115 变为 5/115。目标提示对匹配其他标题在 HotpotQA 有四个配对胜例，对无提示仅净增一道，2Wiki 净少一道。旧 64 输出逐题只精确复现 75/115。暂停加 token、标题提示扩样和查询训练。下一步用 W2a 真实返回直接做冻结作答，以判断取齐证据能否在实际检索噪声和上下文限制下提升回答。[W2b 指南](../docs/factory/RSI_wikipedia_W2b.md)

## 2026-11-18 反馈 29 后验证独立页面的候选动作

W2b 标题诊断真实返回相对匹配无关提升 +25.0、+23.7 个百分点。115 道全匹配且无丢段落。标题臂完整覆盖 93 道，全部题仅答对 33 道，取齐后的作答仍有损失。旧 115 题及其子集保留诊断用途。下一步在支持标题隔离的状态比较一次原题查询与候选排序第一，并量化池中是否有一条可取齐，判别候选构造和选择的限制。[W3a 指南](../docs/factory/RSI_wikipedia_W3_candidates.md)

## 2026-11-18 反馈 30 后审计候选动作的取证上界

W3a 两数据集各 80 道，原索引锚点 8/8 复现。原题和候选第一各取齐 0/160，候选池最佳单条取齐 1/160。目标标题在可见全列表出现 75 道，截断候选出现 5 道。已知标题探针命中任一缺句 55/160，取齐 25/160。W2a 的 104/115 和 93/115 来自要求缓存可回溯的样本，W3a 不需该条件。这限制了旧取证比例的外推，新样本单次标题查询也无法判定语料缺句。

先固定这 160 道做 W3b，审计原语料逐句覆盖，分别测标题截断、问题重排和同标题加问题语境。对缺一句或多句分别按涉及支持页数分层，并检查直接标题探针有效却候选不可达的交集。两阶段使用 CPU 客户端和原检索服务。暂缓扩大候选上线、作答回放和训练，存在双数据集可执行收益后再考虑下一步。正式测试另冻结隔离的支持页。[W3b 指南](../docs/factory/RSI_wikipedia_W3_audit.md)

## 2026-11-18 反馈 31 后验证严格覆盖的含义

W3b 共执行 4,226 次查询，原题和主标题均逐题有序复现，锚点 8/8 通过。原语料下至少一条精确出现 95/160，全部 37/160。一页缺句组取齐 31/63，多页组 6/97。问题重排提高目标标题前三覆盖，两个数据集最佳首选仅各取齐 2/80。前八内的最佳单项写法分别取齐 2/80 和 11/80。主标题拼接整题后，诊断取齐从 10、15 降至 7、7。目标页证据丢在不可见和排名后的计数以命中该页任一缺句为口径。

给同一语料做一次匹配敏感性审计，将同标题相邻分段与异标题的精确定位单列。再让每个已知缺失页各查一次，以同题最佳单条作对照，确定多页题追加查询的诊断空间。暂缓改索引、放宽正式评分、调排序和 GPU 训练。若可用动作空间仍小，选题单位改为语料内可验证的失败修复，基准失败状态保留为全分母诊断。方法对照仍检验相同预算下的独立页迁移。[W3c 指南](../docs/factory/RSI_wikipedia_W3_coverage.md)

## 2026-11-23 选择离线持续强化学习方案 C

Wikipedia 双监督在数据准备门槛处收束，方法训练未开始。当前候选测试在不同任务间连续训练时，能否保留后续任务的成功采样和学习进展。C1 先验证必要性。C2 用两个种子和五个对照检验成功前缀保护。采用公开任务与本地判分器，厂内全程离线。 [冻结协议](protocols/continual_rlvr.md) · [厂内运行指南](../docs/factory/continual_rlvr.md)

## 2026-11-23 修复持续 RLVR 的输入长度校验

首次正式冒烟在模型启动时因 4 道训练提示超限中止，按 INCOMPLETE 记录。临时诊断包运行了真实四卡更新，C1 尚无正式读数。新版使用原 768 token 上限和同一聊天模板，在打包及 CPU 预检时逐条校验。只从原固定保留池替换超限训练题，保留原开发题、锚点、训练规模、生成预算和判定门槛。厂内重建 v2 数据包并完成正式冒烟后运行 C1，GO 后才执行 C2。 [恢复指南](../docs/factory/continual_rlvr.md)

## 2026-11-23 C1 正式 NO-GO 并转入 B0

r2 冒烟 GO，C1 完整结束。A 的 pass@1 从 28.91% 到 43.95%，基座 B 两组 pass@16 为 70.31% 和 69.53%，两组各净减少 3 道成功题，下降均为 1.17 个百分点。按题配对区间为 [−1.56, 3.71] 个百分点。C1 按 5 点冻结门槛判定 NO-GO，C2 未启动。

B0 先检查本地 ALFWorld 文本环境是否能构造可复验的、有实际决策后果的分支。CPU 准入只看输入可行性。 [B0 冻结协议](protocols/decision_check_b0.md) · [厂内运行指南](../docs/factory/decision_check_b0.md)

## 2026-11-24 B0 中断终态与研究收束

厂内反馈显示，`cb243f3` 在 221/240 局中断，212 局未构造出合格的检查候选，9 局通过。两个封存集各有 100 局。剩余 19 局都在同一任务类型，即使全部通过也只有 28 局，最多一类达 10 局。运行判为 INCOMPLETE，研究门槛不可达，不安排补跑或训练。失败分布定位到冻结的分支构造条件，具体筛项的淘汰数量尚未逐项记录。

为旧目录添加只读核对后归档的入口，保留原进度和逐局记录。后续单局独立限时，不改采样、筛选条件和阈值。先复核备选 A 的自然题对照输入，当前无新训练任务。 [收尾指南](../docs/factory/decision_check_b0.md)

## 2026-11-24 确认 B0 完成归档

厂内在 5e490bd 确认旧进程已不存在，直接在原目录写入 INCOMPLETE 归档，原执行提交为 cb243f3，原 221 条记录和封存清单保留。看板实显 INCOMPLETE 221/240，研究门槛仍为 UNREACHABLE。当前无待执行的厂内研究任务。

方案 A 仍待动作对照。W7 的两种补查与补查后材料的呈现比较，无法提供同一首检状态下的重读效果。先验证等预算的检索与重读在逐题结果上有可利用的互补，再考虑控制器。

## 2026-11-24 冻结预算受限强化学习同前缀试点

C1 的冻结 3B 数学基座在 384 token 下截断 349/512 条。新试点从同一未结束前缀采短答与自然长续写，用本地精确判分观察长预算反馈。冻结 256 道与 C1 无重叠的公开数学题，四卡最多两小时，只推理和判分。先过迟到正信号与同成本 RMSE 的双门槛再决定训练。 [冻结协议](protocols/budget_rlvr_paired_pilot.md) · [厂内指南](../docs/factory/budget_rlvr_paired_pilot.md)

## 2026-11-24 同前缀试点正式 NO-GO

2c3cd8e 冒烟通过，2,048 条正式面板完整。可恢复正信号 453 条，成本最大最小比 1.00793。冻结拟方法 paired_disagreement 的条件 RMSE 为 0.0158475，最强对照 full_long 为 0.0079124，误差高 100.29%。反馈正文的 62.64% 属于 short_calibrated_critic 对照。按原冻结门槛停止该候选。采集耗时约 12.5 分钟。对照成本为面板回算期望 token，后续训练表现仍未测。 [冻结协议](protocols/budget_rlvr_paired_pilot.md)

## 2026-11-24 冻结可验证反馈组合首筛

切换至离线算法题。比较同一自然失败在一过一错四格反馈下的独立通过率，用交叉差监督训练选择器。相似度、双失败、重新作答、同数据普通排序是预注册对照。阶段为 CPU 数据检查、四卡冒烟、四卡首筛。真实实验待运行。 [冻结协议](protocols/feedback_pair.md) · [厂内指南](../docs/factory/feedback_pair.md)

## 2026-11-24 补齐反馈组合数据与运行准入

厂内报告对替代执行环境完成了功能探针，缺少只读挂载与真实系统接口等原协议执行条件。加强实测预检，正式判分等待受支持的隔离环境。固定 CodeContests 上游 39 个训练分片的内容摘要，新增纯 CPU 离线验收。数据准备尚未运行，冒烟和正式首筛均无研究结果。样本门槛、方法及对照保持冻结。 [厂内指南](../docs/factory/feedback_pair.md)

## 2026-11-25 替换为内置无 mount 沙箱并封堵反馈 #60 三处缺陷

厂内反馈 #60 确认 39/39 CodeContests 分片已通过 `verify-source`（7,509,753,086 字节，SHA-256 全对，pyarrow 可读 13,328 行），但容器 AppArmor (`cri-containerd.apparmor.d`) 在系统调用层禁用 `mount(2)`，真 `bwrap` 无法启动。同时排查出原沙箱与外部 shim 的三处缺陷：用户命名空间内 uid 0 使 `RLIMIT_NPROC=1` 失效、shim 漏掉 `CLONE_NEWPID` 导致子进程泄漏到宿主机、`timeout` 仅杀父进程不回收子孙。

不放弃真实程序执行、也不申请特权容器或修改 AppArmor，而是将执行器替换为内置的无 mount 沙箱 `work3-userns-chroot-v1`：root 容器下不调用 `mount(2)`，切换为非 root 外部 UID（默认 `65534:65534`）使 `RLIMIT_NPROC=1` 真实生效、显式 `CLONE_NEWPID` 由 PID 1 原子回收、叠加内核 `seccomp-BPF` 拦截进程/线程创建与逃逸系统调用、最小化只读 chroot 根目录与写密封 `memfd`、launcher 自身限时杀私有 PID 1 并配外层 `killpg` 兜底。新增 4 秒有界预检 `probe-sandbox` 独立验证 seccomp (`EPERM`) 与无 seccomp 下的 `RLIMIT_NPROC=1` (`EAGAIN`)。方法、数据拆分、预算与去留门槛保持不变。 [厂内指南](../docs/factory/feedback_pair.md)

## 2026-11-25 放宽私有 netns 对内核默认下线隧道口（tunl0）的断言

厂内反馈 #61 确认 `7c91c28` 无 `mount(2)` 沙箱在厂内容器中正常启动，非 root uid、私有 PID 命名空间、`RLIMIT_NPROC=1`、seccomp 拦 `fork (EPERM)` 与线程拦截均已通过；仅在网络预检因内核加载 `ipip` 模块为每个新 netns 自动创建下线伪接口 `tunl0` 而触发旧的 `names != ['lo']` 名称断言。改为：launcher 在 `unshare` 后直接校验 `/proc/self/ns/net` inode 变化，同时在沙箱内通过 `SIOCGIFFLAGS`/`SIOCGIFADDR` 核验所有允许的回环及内核默认隧道接口均处于 `DOWN` 且无 IPv4 地址，并保留全部连通性探测。研究方法与门槛不变。

## 2026-11-26 停止可验证反馈组合，转向放大差距自博弈 P0a

厂内反馈 #62 确认 `80a45f7` 的 `probe-sandbox` 全部通过，但 `prepare` 在 448/1638 处 INCOMPLETE：外层 `communicate` 的 5.5 秒期限同时计入并发下网络命名空间创建的排队（`--cpu-jobs 8` 时 96 个任务中 69 个超过 5.5 秒，最长 26.72 秒），而 launcher 自身的执行计时本来就在命名空间建好后才开始。

方法复核认为 `feedback_pair_v1` 即使修好执行器也难过门槛，主要依据有四条：

- 冻结 0.5–1.5B 代码模型上，真实失败用例反馈与盲重采样打平（arXiv 2606.31511）；工作二的搜索智能体也有同样结构。
- 四格选择的收益上限低于 5 个百分点的门槛。
- 选择器是手工特征上的岭回归，方法贡献偏薄。
- 方向偏离学位论文主线。

该方向不产生研究 verdict。

工作三改为递归自我提升方向的「放大差距自博弈」：零数据自博弈中，出题奖励改为高低推理预算下的众数一致率差，伪标签取高预算众数。先冻结纯推理 P0a，在 Hendrycks MATH 训练集上检验两点：高预算一致性是否对应正确性；按放大差距选出的题，标签是否比 R-Zero 式按一致率区间选题更准。NO-GO 时转备选「可保证完备性的智能体检索」。

沙箱同时修复：外层期限改为启动预算（默认 60 秒，`WORK3_SANDBOX_STARTUP_SECONDS`）加执行限额加兜底；执行限额（墙钟、CPU、内存）可配置，默认值不变；记录每次调用耗时。 [冻结协议](protocols/amp_gap_p0a.md) · [厂内指南](../docs/factory/amp_gap_p0a.md)

## 2026-11-26 P0a 协议修订 1：采样限制在分词器范围内

厂内反馈 #63：`db4d916` 主模型冒烟 `PASS`（15 题，21 个程序成功执行，`infra` 为 0），正式 P0a 在第一批 INCOMPLETE。三张卡在 `force` 臂报 `Token id 151710/151764/151756 is out of vocabulary`，退出码中的 `None` 是仍在运行、随后被一并停止的第 0 卡。

根因是 Qwen3 的 embedding 有 151,936 行，分词器只认到 151,668。`short` 偶尔采到末尾 267 个未训练保留 id，`force` 臂把 token 原样接回提示时，vLLM 0.8.5 按分词器最大 id 校验，拒收提示。

修订内容：

- 所有臂用 vLLM `allowed_token_ids` 限制在分词器范围内（0.8.5 V1 以 GPU 掩码实现）；
- 每张卡生成前自检掩码是否生效；
- `force` 臂接回 token 前过滤并计数，作为兜底；
- 汇总报告越界计数，冒烟要求越界为 0；
- 提示长度加预算超过上下文时提前报错。

判定门槛、题目抽样、预算、种子和样本数不变。厂内需在新提交下重跑冒烟（`-v2`）后再跑正式 P0a。尚无研究读数。 [冻结协议](protocols/amp_gap_p0a.md) · [厂内指南](../docs/factory/amp_gap_p0a.md)
