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
