#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: MODEL_DIR=/path/to/Qwen2.5-7B-Instruct %s {confirmation|batch|evidence|terminal}\n' "$0"
}
if [[ $# -ne 1 ]]; then usage >&2; exit 2; fi
if [[ $1 == --help || $1 == -h ]]; then usage; exit 0; fi
phase=$1
case "$phase" in
  confirmation) module=work3.rsi_calibrate; phase_args=(--phase confirmation --condition tool_compare) ;;
  batch|evidence) module=work3.rsi_failure_probe; phase_args=(--phase "$phase") ;;
  terminal) module=work3.rsi_terminal_probe; phase_args=() ;;
  *) usage >&2; exit 2 ;;
esac

: "${MODEL_DIR:?Set MODEL_DIR to the local tokenizer/model directory used in the previous run}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$root"
run_dir=${RUN_DIR:-"$root/results/rsi-$phase-$(date +%Y%m%d-%H%M%S)-$$"}
printf 'Revision: %s\nOutput: %s\n' "$(git rev-parse HEAD)" "$run_dir"
exec "${PYTHON:-python}" -m "$module" "${phase_args[@]}" \
  --endpoint "${RSI_ENDPOINT:-http://127.0.0.1:8187}" --model "${RSI_MODEL:-rsi-qwen25-7b}" \
  --tokenizer "$MODEL_DIR" --concurrency "${RSI_CONCURRENCY:-4}" --out-dir "$run_dir"
