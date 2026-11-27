#!/usr/bin/env bash
# Answer-prior kill test (plan A). Guide: docs/factory/answer_prior_p0.md; protocol: research/protocols/answer_prior_p0.md
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${AP_PYTHON:-${AMP_PYTHON:-python3}}" -m work3.answer_prior "$@"
