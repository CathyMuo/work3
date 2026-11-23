#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${REPLAY_PYTHON:-python}" -m work3.wiki_rsi_feedback "$@"
