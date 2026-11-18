#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  audit|prepare)
    export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python="${DIAG_PYTHON:-python}"
    ;;
  smoke|replay)
    python="${REPLAY_PYTHON:-python}"
    ;;
  -h|--help)
    python="${DIAG_PYTHON:-python}"
    ;;
  *)
    printf '%s\n' "Usage: bash scripts/run_wiki_rsi.sh {audit|prepare|smoke|replay} --run NAME" >&2
    exit 2
    ;;
esac
cd "$repo"
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
exec "$python" -m work3.wiki_rsi_replay "$@"
