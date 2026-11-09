#!/usr/bin/env bash
# Work 3 progress board: a read-only local web service (research briefing + live run progress).
#   bash scripts/run_board.sh            -> http://127.0.0.1:8766/
# Reads $DRZERO_DIR/work3_p0, $DRZERO_DIR/work3_p1 and, for P2, $ARM_ROOT/{beta,ncp}.
# Env: BOARD_PYTHON (python), BOARD_HOST (127.0.0.1), BOARD_PORT (8766), DRZERO_DIR, ARM_ROOT,
#      BOARD_P0_OUT / BOARD_P1_OUT (override the run directories).
set -euo pipefail
cd "$(dirname "$0")/.."
py=${BOARD_PYTHON:-python}
if ! command -v "$py" >/dev/null 2>&1; then printf '%s\n' 'Python 不存在；请设置 BOARD_PYTHON。' >&2; exit 1; fi
if [ -z "${DRZERO_DIR:-}" ] && [ -z "${BOARD_P0_OUT:-}" ]; then
  printf '%s\n' '提示：未设置 DRZERO_DIR，看板只显示研究说明。' >&2
fi
exec "$py" -m work3.board --host "${BOARD_HOST:-127.0.0.1}" --port "${BOARD_PORT:-8766}" "$@"
