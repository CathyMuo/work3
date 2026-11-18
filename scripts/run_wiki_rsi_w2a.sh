#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
case "${1:-}" in
  prepare|contract-prepare|budget-prepare)
    export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python="${DIAG_PYTHON:-python}"
    ;;
  run|contract|budget)
    python="${REPLAY_PYTHON:-python}"
    ;;
  *)
    printf '%s\n' "Usage: bash scripts/run_wiki_rsi_w2a.sh {prepare|run|contract-prepare|contract|budget-prepare|budget} --run NAME" >&2
    exit 2
    ;;
esac
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
exec "$python" -m work3.wiki_rsi_w2 "$@"
