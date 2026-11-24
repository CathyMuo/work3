#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py="${BUDGET_PYTHON:-python}"
case "${1:-}" in
  smoke) exec timeout --signal=TERM --kill-after=30s 600s "$py" -m work3.budget_rlvr "$@" ;;
  pilot) exec timeout --signal=TERM --kill-after=30s 7200s "$py" -m work3.budget_rlvr "$@" ;;
  *) exec "$py" -m work3.budget_rlvr "$@" ;;
esac
