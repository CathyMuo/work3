#!/usr/bin/env bash
# Analyze saved P0 evaluation dumps. Uses the active Python environment and CPU only.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
: "${DRZERO_DIR:?Set DRZERO_DIR to the original P0 Dr. Zero checkout}"
drzero="$(cd "$DRZERO_DIR" && pwd)"
out="${OUT_ROOT:-$drzero/work3_p0}"
output="${RETRIEVAL_DIAG_OUT:-$out/retrieval_diag}"
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$repo"
exec "${DIAG_PYTHON:-python}" -m work3.p0_retrieval_diag \
  --eval-root "$out/eval" --test-parquet "$drzero/data/test.parquet" --out-dir "$output" "$@"
