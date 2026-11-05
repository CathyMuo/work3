#!/usr/bin/env bash
# P0: reproduce Dr. Zero Qwen2.5-3B on one 4-GPU node.
# Protocol: research/protocols/P0_drzero_reproduction.md (frozen).
#
# Usage: bash scripts/p0/run_p0.sh <stage>
#   check | prepare | patch | iter1 | iter2 | iter3 | eval0 | eval1 | eval2 | eval3 | gate
#
# Required env:
#   DRZERO_DIR   Dr. Zero checkout at the pinned commit (data/, corpus/, checkpoints/ live here)
#   MODEL_3B     local Qwen2.5-3B-Instruct directory (basename must be Qwen2.5-3B-Instruct)
#   CORPUS_JSONL wiki-18.jsonl            INDEX_PATH  e5_Flat.index
# Optional env:
#   CONDA_ENV (drzero)  NGPU (4)  FAISS_GPU (1; 0 = CPU retrieval)  OUT_ROOT ($DRZERO_DIR/work3_p0)
#   E5_MODEL (intfloat/e5-base-v2; set a local path when offline)
#   GEN_MEM  (unset = upstream 0.8; memory fraction for data generation, only if it OOMs)
set -euo pipefail

PIN=98c4a6dd2495e85d8b4d947ab4c030e540c922c5
RATIO=4321
WORK3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${DRZERO_DIR:?set DRZERO_DIR}" "${MODEL_3B:?set MODEL_3B}"
CONDA_ENV="${CONDA_ENV:-drzero}"
NGPU="${NGPU:-4}"
FAISS_GPU="${FAISS_GPU:-1}"
OUT_ROOT="${OUT_ROOT:-$DRZERO_DIR/work3_p0}"
E5_MODEL="${E5_MODEL:-intfloat/e5-base-v2}"
export DRZERO_DIR MODEL_3B
MODEL_NAME="qwen2.5-3b-instruct"
CKPT="checkpoints/dr-zero"
export PYTHONPATH="$DRZERO_DIR:$WORK3_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OUT_ROOT/logs"
cd "$DRZERO_DIR"

log() { echo "[p0 $(date '+%F %T')] $*"; }

cleanup_servers() {
  pkill -f "search/retrieval_server.py" 2>/dev/null || true
  pkill -f "sglang.launch_server" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  sleep 5
}

timed() {  # timed <name> <cmd...>: run, tee log, record wall-clock seconds
  local name="$1"; shift
  local t0; t0=$(date +%s)
  log "start $name"
  set +e
  "$@" 2>&1 | tee "$OUT_ROOT/logs/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$name $(( $(date +%s) - t0 ))" >> "$OUT_ROOT/timings.txt"
  cleanup_servers
  if [ "$rc" -ne 0 ]; then log "FAILED $name (exit $rc)"; exit "$rc"; fi
  log "done $name"
}

need() { [ -e "$1" ] || { log "missing: $1"; exit 2; }; }

stage_check() {
  [ "$(git rev-parse HEAD)" = "$PIN" ] || { log "Dr. Zero checkout is not at $PIN"; exit 2; }
  [ "$(basename "$MODEL_3B")" = "Qwen2.5-3B-Instruct" ] || { log "MODEL_3B basename must be Qwen2.5-3B-Instruct"; exit 2; }
  need "$MODEL_3B/config.json"
  python - <<'PY'
import importlib, sys
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
import pandas, tensordict, faiss
print("pandas", pandas.__version__, "tensordict", tensordict.__version__, "faiss", faiss.__version__)
if int(pandas.__version__.split(".")[0]) >= 3:
    bad.append("pandas >= 3 breaks upstream main_generation (copy-on-write)")
import verl, os
here = os.path.dirname(verl.__file__)
if not here.startswith(os.environ["DRZERO_DIR"]):
    bad.append(f"verl imported from {here}, not from DRZERO_DIR (uninstall pip verl)")
import torch
print("gpus", torch.cuda.device_count())
if bad:
    print("PROBLEMS:", *bad, sep="\n  ")
    sys.exit(2)
PY
  log "check ok"
}

stage_prepare() {
  need "${CORPUS_JSONL:?set CORPUS_JSONL}"; need "${INDEX_PATH:?set INDEX_PATH}"
  mkdir -p corpus data
  ln -sfn "$CORPUS_JSONL" corpus/wiki-18.jsonl
  ln -sfn "$INDEX_PATH" corpus/e5_Flat.index
  mkdir -p "$OUT_ROOT/patched"
  sed "s#\"Qwen/Qwen2.5-3B-Instruct\"#\"$MODEL_3B\"#" process_train.py > "$OUT_ROOT/patched/process_train.py"
  timed prepare_train python "$OUT_ROOT/patched/process_train.py" --local_dir ./data --corpus_dir ./corpus/wiki-18.jsonl
  timed prepare_test python process_test.py --local_dir ./data
  python - <<'PY'
import pandas as pd
for f in ["data/zero_ratio4321.parquet", "data/test.parquet"]:
    df = pd.read_parquet(f)
    print(f, len(df)); print(df["data_source"].value_counts().to_string())
PY
}

stage_patch() {
  local dst="$OUT_ROOT/patched"
  mkdir -p "$dst"
  local half=$(( NGPU / 2 ))
  for t in 1 2 3; do
    for kind in challenger gen_data solver; do
      local sedargs=(
        -e "s#^gpus=8\$#gpus=$NGPU#"
        -e "s#^dp=4\$#dp=$half#"
        -e "s#^dp=8\$#dp=$NGPU#"
        -e "s#^model=Qwen/Qwen2.5-3B-Instruct\$#model=$MODEL_3B#"
        -e "s#^source activate llm;\$#source \"\$(conda info --base)/etc/profile.d/conda.sh\"; conda activate $CONDA_ENV;#"
        -e "s#--retriever_model='intfloat/e5-base-v2'#--retriever_model='$E5_MODEL'#"
      )
      [ -z "${GEN_MEM:-}" ] || sedargs+=(-e "s#^rollout_memory_utilization=0.8\$#rollout_memory_utilization=$GEN_MEM#")
      [ "$FAISS_GPU" = "1" ] || sedargs+=(-e "/^ *--faiss_gpu \\\\\$/d")
      sed "${sedargs[@]}" "iter${t}_${kind}.sh" > "$dst/iter${t}_${kind}.sh"
    done
  done
  grep -n "^gpus=\|^dp=\|^model=\|conda activate\|faiss_gpu\|retriever_model\|^rollout_memory" "$dst"/iter1_*.sh
  log "patched scripts in $dst"
}

convert_solver() {  # convert_solver <t>
  local base="$CKPT/solver_iter$1_ratio${RATIO}_grpo_group5_${MODEL_NAME}"
  need "$base/global_step_50/actor"
  python -m verl.model_merger merge --backend fsdp \
    --local_dir "$base/global_step_50/actor" --target_dir "$base/solver_iter$1_hf"
}

stage_iter() {  # stage_iter <t>
  local t="$1" p="$OUT_ROOT/patched"
  need "$p/iter${t}_challenger.sh"
  if [ "$t" -gt 1 ]; then
    need "$CKPT/solver_iter$((t-1))_ratio${RATIO}_grpo_group5_${MODEL_NAME}/solver_iter$((t-1))_hf/config.json"
  fi
  cleanup_servers
  timed "iter${t}_challenger" bash "$p/iter${t}_challenger.sh" "$RATIO" "trainer.total_training_steps=$((50 * t))"
  timed "iter${t}_gen_data" bash "$p/iter${t}_gen_data.sh" "$RATIO"
  timed "iter${t}_pool_check" python -m work3.pool_check \
    "data/zero_challenger_iter${t}_ratio${RATIO}_grpo_batch_group1-5_${MODEL_NAME}.parquet" \
    --out "$OUT_ROOT/pool_check_iter${t}.json"
  timed "iter${t}_solver" bash "$p/iter${t}_solver.sh" "$RATIO" "trainer.total_training_steps=50"
  timed "iter${t}_convert" convert_solver "$t"
}

start_retriever() {
  local gpu_flag=()
  [ "$FAISS_GPU" = "1" ] && gpu_flag=(--faiss_gpu)
  python search/retrieval_server.py --index_path=./corpus/e5_Flat.index \
    --corpus_path=./corpus/wiki-18.jsonl --retriever_model="$E5_MODEL" \
    --retriever_name=e5 "${gpu_flag[@]}" --topk 3 > "$OUT_ROOT/logs/retriever_eval.log" 2>&1 &
  for _ in $(seq 1 120); do
    curl -s -o /dev/null -X POST http://127.0.0.1:8000/retrieve \
      -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}' && return 0
    sleep 10
  done
  log "retriever did not come up"; exit 3
}

eval_model() {  # eval_model <t> <model_path>
  local t="$1" model="$2" out="$OUT_ROOT/eval/iter$1"
  need "$model/config.json"; need data/test.parquet
  rm -rf "$out"; mkdir -p "$out"
  cleanup_servers
  start_retriever
  python -m verl.trainer.main_ppo \
    --config-path=./config --config-name=search_multiturn_grpo \
    data.train_files=./data/test.parquet data.val_files=./data/test.parquet \
    data.train_batch_size=256 data.max_prompt_length=512 \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path=./config/search_tool_config.yaml \
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
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
