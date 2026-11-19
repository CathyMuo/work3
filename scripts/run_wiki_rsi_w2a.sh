#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
case "${1:-}" in
  prepare|contract-prepare|budget-prepare|answer-prepare|candidates-prepare|candidates|candidate-audit-prepare|candidate-audit|coverage-audit-prepare|coverage-audit|repair-pool-prepare|repair-pool|repair-audit|repair-answer-prepare|natural-prepare|natural-check|natural-query-prepare|action-pilot-prepare|action-pilot|action-train-prepare|overnight-prepare)
    export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python="${DIAG_PYTHON:-python}"
    ;;
  run|contract|budget|answer|repair-answer|natural-query|action-train|overnight)
    python="${REPLAY_PYTHON:-python}"
    ;;
  *)
    printf '%s\n' "Usage: bash scripts/run_wiki_rsi_w2a.sh PHASE --run NAME (see the factory guide for supported phases)" >&2
    exit 2
    ;;
esac
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
exec "$python" -m work3.wiki_rsi_w2 "$@"
