#!/usr/bin/env bash
# P0: reproduce Dr. Zero Qwen2.5-3B on one 4-GPU node.
# Protocol: research/protocols/P0_drzero_reproduction.md (frozen; see its addenda).
#
# Usage: bash scripts/p0/run_p0.sh <stage>
#   check | prepare | patch | iter1 | iter2 | iter3 | eval0 | eval1 | eval2 | eval3 | gate | cleanup
#
# Required env:
#   DRZERO_DIR    Dr. Zero checkout at the pinned commit (data/, corpus/, checkpoints/ live here)
#   MODEL_3B      local Qwen2.5-3B-Instruct directory (basename must be Qwen2.5-3B-Instruct)
#   CORPUS_JSONL  wiki-18.jsonl            INDEX_PATH   e5_Flat.index
#   TRAIN_PARQUET / TEST_PARQUET  local PeterJinGo/nq_hotpotqa_train {train,test}.parquet
#                 (when set, prepare never contacts HuggingFace)
# Optional env:
#   CONDA_ENV (drzero)  NGPU (4)  FAISS_GPU (1; 0 = CPU retrieval)  OUT_ROOT ($DRZERO_DIR/work3_p0)
#   E5_MODEL  (intfloat/e5-base-v2; set a local path when offline)
#   RET_PORT (8100) retriever port      REWARD_PORT (8101) proposer-reward SGLang port
#   GEN_MEM   (unset = upstream 0.8; memory fraction for data generation, only if it OOMs)
#   ALLOW_BUSY_GPU (0; 1 = let `check` pass while other GPU processes are running)
set -euo pipefail

PIN=98c4a6dd2495e85d8b4d947ab4c030e540c922c5
RATIO=4321
TAG="work3-p0"   # every process started by a stage carries WORK3_P0_TAG=$TAG in its environment
WORK3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DRZERO_DIR:?set DRZERO_DIR}" "${MODEL_3B:?set MODEL_3B}"
CONDA_ENV="${CONDA_ENV:-drzero}"
NGPU="${NGPU:-4}"
FAISS_GPU="${FAISS_GPU:-1}"
OUT_ROOT="${OUT_ROOT:-$DRZERO_DIR/work3_p0}"
E5_MODEL="${E5_MODEL:-intfloat/e5-base-v2}"
RET_PORT="${RET_PORT:-8100}"
REWARD_PORT="${REWARD_PORT:-8101}"
export DRZERO_DIR MODEL_3B
MODEL_NAME="qwen2.5-3b-instruct"
CKPT="checkpoints/dr-zero"
PATCHED="$OUT_ROOT/patched"
export PYTHONPATH="$DRZERO_DIR:$WORK3_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OUT_ROOT/logs"
cd "$DRZERO_DIR"

log() { echo "[p0 $(date '+%F %T')] $*"; }
need() { [ -e "$1" ] || { log "missing: $1"; exit 2; }; }

# Kill only processes started by this script's stages (tagged environment).
# Never touches other projects' retrievers, SGLang servers or Ray clusters.
cleanup_servers() {
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
    print(f"[p0 cleanup] {sig.name} {len(pids)} tagged processes", flush=True)
    for p in pids:
        try:
            os.kill(p, sig)
        except OSError:
            pass
    time.sleep(10)
left = tagged()
if left:
    print(f"[p0 cleanup] still alive: {left}", flush=True)
PY
}

timed() {  # timed <name> <cmd...>: run tagged, tee log, record wall-clock seconds, clean up
  local name="$1"; shift
  local t0; t0=$(date +%s)
  log "start $name"
  set +e
  ( export WORK3_P0_TAG="$TAG"; "$@" ) 2>&1 | tee "$OUT_ROOT/logs/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$name $(( $(date +%s) - t0 ))" >> "$OUT_ROOT/timings.txt"
  cleanup_servers
  if [ "$rc" -ne 0 ]; then log "FAILED $name (exit $rc)"; exit "$rc"; fi
  log "done $name"
}

port_free() {  # port_free <port>: exit 0 if nothing listens on it
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
  [ "$(basename "$MODEL_3B")" = "Qwen2.5-3B-Instruct" ] || { log "MODEL_3B basename must be Qwen2.5-3B-Instruct"; bad=1; }
  [ -f "$MODEL_3B/config.json" ] || { log "missing $MODEL_3B/config.json"; bad=1; }
  for p in "$RET_PORT" "$REWARD_PORT"; do
    port_free "$p" || { log "port $p is in use; set RET_PORT / REWARD_PORT to free ports"; bad=1; }
  done
  log "GPU compute processes (should be empty: stop the keep-alive job and its ray workers first):"
  local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true)
  echo "${apps:-  (none)}"
  if [ -n "$apps" ] && [ "${ALLOW_BUSY_GPU:-0}" != "1" ]; then
    log "GPUs are not free (set ALLOW_BUSY_GPU=1 only if these are intentionally kept)"; bad=1
  fi
  python - <<'PY' || bad=1
import importlib, os, sys
want = {"verl": None, "sglang": "0.4.6.post5", "torch": "2.6.0"}
bad = []
for mod, ver in want.items():
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"{mod} {v}")
        if ver and not str(v).startswith(ver):
            bad.append(f"{mod} {v} != {ver}")
    except Exception as e:
        bad.append(f"{mod}: {e}")
import pandas, tensordict, faiss, torch, verl
print("pandas", pandas.__version__, "tensordict", tensordict.__version__, "faiss", faiss.__version__)
if int(pandas.__version__.split(".")[0]) >= 3:
    bad.append("pandas >= 3 breaks upstream main_generation (copy-on-write)")
here = os.path.dirname(verl.__file__)
if not here.startswith(os.environ["DRZERO_DIR"]):
    bad.append(f"verl imported from {here}, not from DRZERO_DIR (uninstall pip verl)")
print("gpus", torch.cuda.device_count())
if bad:
    print("PROBLEMS:", *bad, sep="\n  ")
    sys.exit(2)
PY
  [ "$bad" = 0 ] || { log "check FAILED"; exit 2; }
  log "check ok"
}

stage_prepare() {
  need "${CORPUS_JSONL:?set CORPUS_JSONL}"; need "${INDEX_PATH:?set INDEX_PATH}"
  mkdir -p corpus data "$PATCHED"
  ln -sfn "$CORPUS_JSONL" corpus/wiki-18.jsonl
  ln -sfn "$INDEX_PATH" corpus/e5_Flat.index
  sed "s#\"Qwen/Qwen2.5-3B-Instruct\"#\"$MODEL_3B\"#" process_train.py > "$PATCHED/process_train.py"
  local hf_train=() hf_test=()
  if [ -n "${TRAIN_PARQUET:-}" ]; then
    need "$TRAIN_PARQUET"; hf_train=(-m work3.p0_launch offline-hf --map "train.parquet=$TRAIN_PARQUET" --)
  fi
  if [ -n "${TEST_PARQUET:-}" ]; then
    need "$TEST_PARQUET"; hf_test=(-m work3.p0_launch offline-hf --map "test.parquet=$TEST_PARQUET" --)
  fi
  rm -f data/zero_ratio4321.parquet data/test.parquet data/test_1200.parquet
  timed prepare_train python "${hf_train[@]}" "$PATCHED/process_train.py" --local_dir ./data --corpus_dir ./corpus/wiki-18.jsonl
  timed prepare_test python "${hf_test[@]}" process_test.py --local_dir ./data
  # upstream process_test.py only logs download errors, so check the outputs explicitly
  need data/zero_ratio4321.parquet; need data/test.parquet
  timed prepare_val python -m work3.val_subset data/test.parquet data/test_1200.parquet
  python - <<'PY'
import pandas as pd
for f in ["data/zero_ratio4321.parquet", "data/test.parquet", "data/test_1200.parquet"]:
    df = pd.read_parquet(f)
    print(f, len(df)); print(df["data_source"].value_counts().to_string())
PY
}

stage_patch() {
  mkdir -p "$PATCHED"
  local half=$(( NGPU / 2 ))
  sed "s#http://127.0.0.1:8000/retrieve#http://127.0.0.1:$RET_PORT/retrieve#" \
    config/search_tool_config.yaml > "$PATCHED/search_tool_config.yaml"
  for t in 1 2 3; do
    for kind in challenger gen_data solver; do
      local sedargs=(
        -e "s#^gpus=8\$#gpus=$NGPU#"
        -e "s#^dp=4\$#dp=$half#"
        -e "s#^dp=8\$#dp=$NGPU#"
        -e "s#^model=Qwen/Qwen2.5-3B-Instruct\$#model=$MODEL_3B#"
        -e "s#^source activate llm;\$#source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate $CONDA_ENV;#"
        -e "s#--retriever_model='intfloat/e5-base-v2'#--retriever_model='$E5_MODEL'#"
        -e "s#^kill -9 \$(lsof -t -i :800[01]);\$#: \# port cleanup handled by run_p0.sh#"
        -e "s#^python search/retrieval_server.py \\\\\$#python -m work3.p0_launch serve --port $RET_PORT -- search/retrieval_server.py \\\\#"
        -e "s#^TOOL_CONFIG=\"\$CONFIG_PATH/search_tool_config.yaml\"\$#TOOL_CONFIG=\"$PATCHED/search_tool_config.yaml\"#"
        -e "s#--port=8001 \\\\\$#--port=$REWARD_PORT \\\\#"
        -e "s#base_url=\"http://127.0.0.1:8001\"#base_url=\"http://127.0.0.1:$REWARD_PORT\"#"
        -e "s#^    --topk 3 &\$#    --topk 3 \\&\\nbash $WORK3_DIR/scripts/p0/wait_retriever.sh $RET_PORT || exit 3#"
      )
      [ "$FAISS_GPU" = "1" ] || sedargs+=(-e "/^ *--faiss_gpu \\\\\$/d")
      [ -z "${GEN_MEM:-}" ] || sedargs+=(-e "s#^rollout_memory_utilization=0.8\$#rollout_memory_utilization=$GEN_MEM#")
      sed "${sedargs[@]}" "iter${t}_${kind}.sh" > "$PATCHED/iter${t}_${kind}.sh"
    done
  done
  # every upstream port reference must be gone
  if grep -n "800[01]\|lsof" "$PATCHED"/iter*.sh "$PATCHED/search_tool_config.yaml"; then
    log "patch FAILED: upstream port references left"; exit 2
  fi
  local n_wait; n_wait=$(grep -l "wait_retriever.sh" "$PATCHED"/iter*.sh | wc -l)
  [ "$n_wait" = 9 ] || { log "patch FAILED: retriever wait inserted in $n_wait of 9 scripts"; exit 2; }
  grep -n "^gpus=\|^dp=\|^model=\|conda activate\|faiss_gpu\|retriever_model\|^rollout_memory\|p0_launch\|wait_retriever\|TOOL_CONFIG=\|--port=\|base_url" "$PATCHED"/iter1_*.sh
  log "patched scripts in $PATCHED"
}

convert_solver() {  # convert_solver <t>
  local base="$CKPT/solver_iter$1_ratio${RATIO}_grpo_group5_${MODEL_NAME}"
  need "$base/global_step_50/actor"
  python -m verl.model_merger merge --backend fsdp \
    --local_dir "$base/global_step_50/actor" --target_dir "$base/solver_iter$1_hf"
}

stage_iter() {  # stage_iter <t>
  local t="$1"
  need "$PATCHED/iter${t}_challenger.sh"; need data/test_1200.parquet
  if [ "$t" -gt 1 ]; then
    need "$CKPT/solver_iter$((t-1))_ratio${RATIO}_grpo_group5_${MODEL_NAME}/solver_iter$((t-1))_hf/config.json"
  fi
  cleanup_servers
  timed "iter${t}_challenger" bash "$PATCHED/iter${t}_challenger.sh" "$RATIO" "trainer.total_training_steps=$((50 * t))"
  timed "iter${t}_gen_data" bash "$PATCHED/iter${t}_gen_data.sh" "$RATIO"
  timed "iter${t}_pool_check" python -m work3.pool_check \
    "data/zero_challenger_iter${t}_ratio${RATIO}_grpo_batch_group1-5_${MODEL_NAME}.parquet" \
    --out "$OUT_ROOT/pool_check_iter${t}.json"
  timed "iter${t}_solver" bash "$PATCHED/iter${t}_solver.sh" "$RATIO" "trainer.total_training_steps=50"
  timed "iter${t}_convert" convert_solver "$t"
}

start_retriever() {
  local gpu_flag=()
  [ "$FAISS_GPU" = "1" ] && gpu_flag=(--faiss_gpu)
  python -m work3.p0_launch serve --port "$RET_PORT" -- search/retrieval_server.py \
    --index_path=./corpus/e5_Flat.index --corpus_path=./corpus/wiki-18.jsonl \
    --retriever_model="$E5_MODEL" --retriever_name=e5 "${gpu_flag[@]}" --topk 3 \
    > "$OUT_ROOT/logs/retriever_eval.log" 2>&1 &
  bash "$WORK3_DIR/scripts/p0/wait_retriever.sh" "$RET_PORT"
}

eval_model() {  # eval_model <t> <model_path>
  local t="$1" model="$2" out="$OUT_ROOT/eval/iter$1"
  need "$model/config.json"; need data/test.parquet; need "$PATCHED/search_tool_config.yaml"
  rm -rf "$out"; mkdir -p "$out"
  start_retriever || return 3
  python -m verl.trainer.main_ppo \
    --config-path=./config --config-name=search_multiturn_grpo \
    data.train_files=./data/test.parquet data.val_files=./data/test.parquet \
    data.train_batch_size=256 data.val_batch_size=256 data.max_prompt_length=512 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$model" \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PATCHED/search_tool_config.yaml" \
    trainer.logger='["console"]' trainer.project_name=work3-p0 \
    trainer.experiment_name="eval_iter$t" \
    trainer.n_gpus_per_node="$NGPU" trainer.nnodes=1 \
    trainer.val_before_train=True trainer.val_only=True \
    trainer.validation_data_dir="$out"
}

stage_eval() {  # stage_eval <t>
  local t="$1" model
  if [ "$t" = 0 ]; then model="$MODEL_3B"
  else model="$DRZERO_DIR/$CKPT/solver_iter${t}_ratio${RATIO}_grpo_group5_${MODEL_NAME}/solver_iter${t}_hf"; fi
  cleanup_servers
  timed "eval${t}" eval_model "$t" "$model"
}

stage_gate() {
  local args=()
  for t in 0 1 2 3; do
    [ -d "$OUT_ROOT/eval/iter$t" ] && args+=("--iter$t" "$OUT_ROOT/eval/iter$t")
  done
  python - "$OUT_ROOT" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
files = sorted(root.glob("pool_check_iter*.json"))
ok = bool(files) and all(json.loads(f.read_text())["ok"] for f in files)
(root / "pool_check_all.json").write_text(json.dumps({"ok": ok, "files": [str(f) for f in files]}))
PY
  python -m work3.p0_gate "${args[@]}" --pool-check "$OUT_ROOT/pool_check_all.json" \
    --out "$OUT_ROOT/p0_summary.json" || true
  log "summary: $OUT_ROOT/p0_summary.json ; timings: $OUT_ROOT/timings.txt"
}

case "${1:-}" in
  check) stage_check ;;
  prepare) stage_prepare ;;
  patch) stage_patch ;;
  iter1|iter2|iter3) stage_iter "${1#iter}" ;;
  eval0|eval1|eval2|eval3) stage_eval "${1#eval}" ;;
  gate) stage_gate ;;
  cleanup) cleanup_servers ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
