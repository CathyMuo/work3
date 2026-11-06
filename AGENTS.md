# Work 3 development rules

Independent project rooted at this directory. Work 3 is a **method** paper
(direction under re-scoping since 2026-11-05: the PBRO reflection-credit idea was
dropped; the new candidate area is zero-data self-evolving search agents).
It contains no GMST methods or artifacts. Do not modify Work 2 (`../work2_byd`)
or the stopped projects elsewhere in the workspace; Work 2 code may be copied in
only with its source commit recorded.

Status (2026-11-06): P0 and P1 gates frozen; no run registered yet. Read `research/EXPERIMENT_PLAN.md`,
`research/STATE.json` and `research/DECISION_LOG.md` before changing research code
or proposing experiments. Pilots P0/P1/P2 must have frozen gates committed before
their data are seen; gates are never changed after results.

Target venue: ICML 2027 (primary), SIGIR 2027 (alternative); decide by 2026-12-15.

## Collaboration (same as Work 2)

- Develop, test (CPU) and commit on the development server (xd-5090); push to the public
  repository https://github.com/CathyMuo/work3 (user authorized public, 2026-11-05;
  renamed from work3_byd and fully public including method code, 2026-11-06).
- The user pulls over HTTPS and runs on the 4×H800 factory server, then pastes back
  summaries. Never require the factory to push, hold write credentials, or edit
  `research/DECISION_LOG.md` / `research/STATE.json`.
- The factory environment is user-managed. Provide recommended configuration only.
- Keep paths configurable; models, corpora, indexes, outputs and credentials are
  machine-local and never committed (`configs/local.json` is ignored).
- Record main commit, pinned dependency commits, config and model/index versions with
  every run. Do not invent results; report factory execution as pending until real
  results are supplied.
- `third_party/` dependencies are pinned and unmodified; patches live in this repo.
  Dr. Zero is cloned beside the repo at the commit in `third_party/PLANNED_UPSTREAMS.md`;
  its launch scripts are adapted only by `scripts/p0/run_p0.sh`.
- Run tests with the factory-equivalent Python
  (`/mnt/wh/miniconda3/envs/searchr1/bin/python -m unittest discover -s tests -q`);
  import shared test helpers as `from test_x import ...`, never `from tests.x`.
