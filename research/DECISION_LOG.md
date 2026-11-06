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
