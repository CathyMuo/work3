#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
: "${DRZERO_DIR:?Set DRZERO_DIR to the original P0 Dr. Zero checkout}"
drzero="$(cd "$DRZERO_DIR" && pwd)"
run_root="${OUT_ROOT:-$drzero/work3_p0}"
output="${WIKI_RSI_OUT:-$run_root/wiki_rsi_audit-$(date +%Y%m%d-%H%M%S)-$$}"
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$repo"
exec "${DIAG_PYTHON:-python}" -m work3.wiki_rsi_audit --eval-root "$run_root/eval" \
  --test-parquet "$drzero/data/test.parquet" --out-dir "$output" "$@"
