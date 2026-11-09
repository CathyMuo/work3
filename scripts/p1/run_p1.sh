#!/usr/bin/env bash
# P1: noise diagnosis on the P0 pools. Protocol: research/protocols/P1_noise_diagnosis.md
# (frozen; see its addendum). Run only after P0 has finished and been reported.
#
# Usage: bash scripts/p1/run_p1.sh <stage>
#   check | pool1 | pool3 | parity | summary | cleanup
#
# Required env (same values as P0): DRZERO_DIR, MODEL_3B
# Optional env: CONDA_ENV (drzero) NGPU (4) FAISS_GPU (1) E5_MODEL RET_PORT (8100) REWARD_PORT (8101)
#   P0_OUT ($DRZERO_DIR/work3_p0)  P1_OUT ($DRZERO_DIR/work3_p1)  P1_MEM (0.6)  P1_CONCURRENCY (64)
#   ALLOW_BUSY_GPU (0)
set -euo pipefail

PIN=98c4a6dd2495e85d8b4d947ab4c030e540c922c5
TAG="work3-p1"
WORK3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DRZERO_DIR:?set DRZERO_DIR}" "${MODEL_3B:?set MODEL_3B}"
NGPU="${NGPU:-4}"
FAISS_GPU="${FAISS_GPU:-1}"
E5_MODEL="${E5_MODEL:-intfloat/e5-base-v2}"
RET_PORT="${RET_PORT:-8100}"
REWARD_PORT="${REWARD_PORT:-8101}"
P0_OUT="${P0_OUT:-$DRZERO_DIR/work3_p0}"
P1_OUT="${P1_OUT:-$DRZERO_DIR/work3_p1}"
P1_MEM="${P1_MEM:-0.6}"
P1_CONCURRENCY="${P1_CONCURRENCY:-64}"
export DRZERO_DIR MODEL_3B
export PYTHONPATH="$DRZERO_DIR:$WORK3_DIR:${PYTHONPATH:-}"
RATIO=4321
MODEL_NAME="qwen2.5-3b-instruct"
CKPT="$DRZERO_DIR/checkpoints/dr-zero"
TOOL_CFG="$DRZERO_DIR/config/search_tool_config.yaml"   # read only for the tool schema

mkdir -p "$P1_OUT/logs"
export WORK3_RETRIEVER_STATS="$P1_OUT/logs/retriever_stats.json"
cd "$DRZERO_DIR"

log() { echo "[p1 $(date '+%F %T')] $*"; }
need() { [ -e "$1" ] || { log "missing: $1"; exit 2; }; }
pool_file() { echo "$DRZERO_DIR/data/zero_challenger_iter$1_ratio${RATIO}_grpo_batch_group1-5_${MODEL_NAME}.parquet"; }
pool_model() {  # reward solver the proposer was trained against
  if [ "$1" = 1 ]; then echo "$MODEL_3B"
  else echo "$CKPT/solver_iter2_ratio${RATIO}_grpo_group5_${MODEL_NAME}/solver_iter2_hf"; fi
}

cleanup_servers() {  # kill only processes carrying WORK3_P0_TAG=$TAG
  python3 - "$TAG" "$$" <<'PY'
import os, signal, sys, time
tag, me = ("WORK3_P0_TAG=" + sys.argv[1]).encode(), int(sys.argv[2])
def tagged():
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in (me, os.getpid(), os.getppid()):
            continue
        try:
            with open(f"/proc/{d}/environ", "rb") as f:
                if tag in f.read().split(b"\0"):
                    out.append(int(d))
        except OSError:
            pass
    return out
for sig in (signal.SIGTERM, signal.SIGKILL):
    pids = tagged()
    if not pids:
        break
    print(f"[p1 cleanup] {sig.name} {len(pids)} tagged processes", flush=True)
    for p in pids:
        try:
            os.kill(p, sig)
        except OSError:
            pass
    time.sleep(10)
PY
}

timed() {
  local name="$1"; shift
  local t0; t0=$(date +%s)
  log "start $name"
  set +e
  ( export WORK3_P0_TAG="$TAG"; "$@" ) 2>&1 | tee "$P1_OUT/logs/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$name $(( $(date +%s) - t0 )) $rc $(date +%s)" >> "$P1_OUT/timings.txt"   # name seconds exit end
  cleanup_servers
  if [ "$rc" -ne 0 ]; then log "FAILED $name (exit $rc)"; exit "$rc"; fi
  log "done $name"
}

port_free() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

stage_check() {
  local bad=0
  [ "$(git rev-parse HEAD)" = "$PIN" ] || { log "Dr. Zero checkout is not at $PIN"; bad=1; }
  for p in 1 3; do
    [ -f "$(pool_file $p)" ] || { log "missing pool: $(pool_file $p)"; bad=1; }
    [ -f "$(pool_model $p)/config.json" ] || { log "missing model: $(pool_model $p)"; bad=1; }
  done
  [ -f "$P0_OUT/pool_check_all.json" ] || { log "missing $P0_OUT/pool_check_all.json (run P0 gate first)"; bad=1; }
  for p in "$RET_PORT" "$REWARD_PORT"; do
    port_free "$p" || { log "port $p is in use"; bad=1; }
  done
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true)
  echo "GPU compute processes: ${apps:-(none)}"
  if [ -n "$apps" ] && [ "${ALLOW_BUSY_GPU:-0}" != "1" ]; then log "GPUs are not free"; bad=1; fi
  python - <<'PY' || bad=1
import os, sys
import sglang, verl
print("sglang", sglang.__version__)
here = os.path.dirname(verl.__file__)
if not here.startswith(os.environ["DRZERO_DIR"]):
    print("verl imported from", here); sys.exit(2)
from verl.workers.rollout.schemas import AsyncRolloutRequest  # needed by parity
PY
  [ "$bad" = 0 ] || { log "check FAILED"; exit 2; }
  log "check ok"
}

serve_and_run() {  # serve_and_run <pool_id>
  local p="$1" model; model="$(pool_model "$p")"
  local gpu_flag=()
  [ "$FAISS_GPU" = "1" ] && gpu_flag=(--faiss_gpu)
  python -m work3.p0_launch serve --port "$RET_PORT" -- search/retrieval_server.py \
    --index_path=./corpus/e5_Flat.index --corpus_path=./corpus/wiki-18.jsonl \
    --retriever_model="$E5_MODEL" --retriever_name=e5 "${gpu_flag[@]}" --topk 3 \
    > "$P1_OUT/logs/retriever_pool$p.log" 2>&1 &
  bash "$WORK3_DIR/scripts/p0/wait_retriever.sh" "$RET_PORT" || return 3
  python -m sglang.launch_server --model-path="$model" --served-model-name=p1-solver \
    --port="$REWARD_PORT" --mem-fraction-static="$P1_MEM" --dp-size="$NGPU" --tp-size=1 \
    --log-level=error > "$P1_OUT/logs/sglang_pool$p.log" 2>&1 &
  local t0; t0=$(date +%s)
  until curl -s -f -o /dev/null "http://127.0.0.1:$REWARD_PORT/v1/models"; do
    [ $(( $(date +%s) - t0 )) -lt 1800 ] || { log "SGLang not ready"; return 3; }
    sleep 10
  done
  log "SGLang ready on :$REWARD_PORT after $(( $(date +%s) - t0 )) s (model $model)"
  # same tokenizer / chat template for both pools (solver checkpoints are fine-tuned from MODEL_3B)
  python -m work3.p1_run --pool-file "$(pool_file "$p")" --pool-id "$p" --tokenizer "$MODEL_3B" \
    --gen-url "http://127.0.0.1:$REWARD_PORT" --model-name p1-solver \
    --search-url "http://127.0.0.1:$RET_PORT/retrieve" --out-dir "$P1_OUT" \
    --tool-config "$TOOL_CFG" --concurrency "$P1_CONCURRENCY"
}

stage_pool() {
  cleanup_servers
  timed "pool$1" serve_and_run "$1"
}

stage_parity() {
  timed parity python -m work3.p1_parity --tokenizer "$MODEL_3B" --tool-config "$TOOL_CFG" \
    --questions "$P1_OUT/pool1_questions.jsonl" "$P1_OUT/pool3_questions.jsonl" \
    --limit 3000 --out "$P1_OUT/p1_parity.json"
}

stage_summary() {
  python -m work3.p1_summary --run-dir "$P1_OUT" --pool-check "$P0_OUT/pool_check_all.json" \
    --parity "$P1_OUT/p1_parity.json" --out "$P1_OUT/p1_summary.json" || true
  log "summary: $P1_OUT/p1_summary.json"
}

case "${1:-}" in
  check) stage_check ;;
  pool1) stage_pool 1 ;;
  pool3) stage_pool 3 ;;
  parity) stage_parity ;;
  summary) stage_summary ;;
  cleanup) cleanup_servers ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
