#!/usr/bin/env bash
# P0: reproduce Dr. Zero Qwen2.5-3B on one 4-GPU node.
# Protocol: research/protocols/P0_drzero_reproduction.md (frozen; see its addenda).
#
# Usage: bash scripts/p0/run_p0.sh <stage>
#   check | prepare | patch | iter1 | iter2 | iter3 | eval0 | eval1 | eval2 | eval3 | gate | cleanup | probe
#   all   (unattended: iter1 eval1 iter2 eval2 iter3 eval3 eval0 gate; finished stages are skipped,
#          a failed stage is retried once from where it stopped -- MAX_ATTEMPTS, ALL_STAGES)
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
#   P0_REWARD_ARM (unset = upstream Dr. Zero reward; beta | ncp = P2 arm, set by scripts/p2/run_p2.sh)
#   CHAL_ROLLOUT_MEM (0.25 = upstream; proposer in-trainer SGLang memory fraction. 0.45 (r2-r5) caused
#                     an OOM at proposer step 16 on GPU 0 and gave no speed-up -- do not raise it)
#   SAVE_FREQ (10; checkpoint every N steps, upstream 25)  CKPT_KEEP (2; checkpoints kept per run)
#   FORCE (0; 1 = re-run sub-stages of an iteration even if their outputs already exist)
#   TRAIN_MICRO (1; per-GPU micro-batch for the actor update and log-prob passes, upstream 2.
#                Same mini-batch and optimiser step; halves the per-pass token count, which is
#                what grew until the OOM at proposer step 30 -- factory feedback #6)
#   EXPANDABLE_SEGMENTS (0; 1 = PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for all stages,
#                        a fragmentation fallback; not verified with SGLang's memory saver)
#   EVAL_ROLLOUT_MEM (0.6; evaluation SGLang memory fraction; memory only)
#   ROLLOUT_TP (1; in-trainer SGLang tensor parallel for proposer and solver, upstream 2; parallelism only)
#   REWARD_TP (2) / REWARD_DP (NGPU/REWARD_TP): proposer-reward SGLang server layout (upstream tp=2)
#   REWARD_METRICS (1; start the reward server with --enable-metrics for work3.speed_probe / the board)
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
CHAL_ROLLOUT_MEM="${CHAL_ROLLOUT_MEM:-0.25}"
SAVE_FREQ="${SAVE_FREQ:-10}"
CKPT_KEEP="${CKPT_KEEP:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
REWARD_TP="${REWARD_TP:-2}"
REWARD_DP="${REWARD_DP:-$(( NGPU / REWARD_TP ))}"
REWARD_METRICS="${REWARD_METRICS:-1}"
export DRZERO_DIR MODEL_3B
MODEL_NAME="qwen2.5-3b-instruct"
CKPT="checkpoints/dr-zero"
PATCHED="$OUT_ROOT/patched"
export PYTHONPATH="$DRZERO_DIR:$WORK3_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
TRAIN_MICRO="${TRAIN_MICRO:-1}"
if [ "${EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
ARM="${P0_REWARD_ARM:-}"
case "$ARM" in ""|beta|ncp) ;; *) echo "P0_REWARD_ARM must be empty, beta or ncp"; exit 2 ;; esac
[ -z "$ARM" ] || export WORK3_REWARD_LOG="$OUT_ROOT/reward_log.jsonl"
export WORK3_RETRIEVER_STATS="$OUT_ROOT/logs/retriever_stats.json"   # written by work3.p0_launch serve

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
# GPU processes this tag does not cover (e.g. a leftover SGLang from an older run) would eat the
# memory the next stage needs. Report them; never kill them, they may belong to someone else.
import subprocess
try:
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30).stdout
except Exception:
    q = ""
for line in q.splitlines():
    try:
        pid, mib = (int(x) for x in line.split(","))
    except ValueError:
        continue
    if mib < 4096 or pid in left:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = "?"
    if any(k in cmd for k in ("sglang", "ray::", "verl", "retrieval_server", "p0_launch", "drzero")):
        print(f"[p0 cleanup] WARNING untagged GPU process pid={pid} {mib / 1024:.0f} GB: {cmd[:160]}", flush=True)
        print("[p0 cleanup]   not started under this run's tag; if it is a leftover of Work 3, kill it by PID before training", flush=True)
PY
}

timed_rc() {  # timed_rc <name> <cmd...>: run tagged, log, record wall-clock seconds, clean up; returns the exit code
  local name="$1"; shift
  local t0; t0=$(date +%s)
  STAGE_T0="$t0"   # outputs accepted after a nonzero exit must be written after this
  local logf="$OUT_ROOT/logs/$name.log"
  log "start $name"
  # The stage writes straight to its log file and the console follows it with `tail -f`. No pipe:
  # upstream scripts leave servers running in the background, and with `cmd | tee` those servers
  # held the pipe open, so the stage never returned after its script had ended (feedback #7-#9).
  : > "$logf"
  tail -n +1 -f "$logf" &
  local tailpid=$!
  set +e
  ( exec 8>&-; export WORK3_P0_TAG="$TAG"; "$@" ) >> "$logf" 2>&1 < /dev/null
  local rc=$?
  set -e
  sleep 1; kill "$tailpid" 2>/dev/null || true; wait "$tailpid" 2>/dev/null || true
  echo "$name $(( $(date +%s) - t0 )) $rc $(date +%s)" >> "$OUT_ROOT/timings.txt"   # name seconds exit end
  [ "$rc" -eq 0 ] || exit_diag "$rc" >> "$OUT_ROOT/logs/$name.log" 2>&1
  cleanup_servers
  return "$rc"
}

exit_diag() {  # exit_diag <rc>: who might have killed it? (appended to the stage log)
  echo "[p0 diag] exit code $1$( [ "$1" -gt 128 ] && echo " = signal $(( $1 - 128 ))" ) at $(date '+%F %T')"
  local cg=/sys/fs/cgroup
  for f in memory.events memory.current memory.max memory.peak; do
    [ -r "$cg/$f" ] && echo "[p0 diag] cgroup $f: $(tr '\n' ' ' < "$cg/$f")"
  done
  dmesg -T 2>/dev/null | grep -iE "killed process|out of memory|oom" | tail -n 5 | sed 's/^/[p0 diag] dmesg: /'
  free -g 2>/dev/null | sed 's/^/[p0 diag] /'
  return 0
}

take_lock() {  # one run_p0.sh per output directory (children of `all` share their parent's lock)
  [ "${WORK3_P0_LOCK_HELD:-0}" = "1" ] && return 0
  command -v flock > /dev/null || return 0
  mkdir -p "$OUT_ROOT"
  exec 8> "$OUT_ROOT/.run_p0.lock"
  if ! flock -n 8; then
    log "another run_p0.sh is already running on $OUT_ROOT; stop it first (bash $0 cleanup stops it)"
    exit 4
  fi
  export WORK3_P0_LOCK_HELD=1
}

stop_orchestrators() {  # stop other run_p0.sh stage runs on this DRZERO_DIR before killing their stages
  python3 - "$$" "$DRZERO_DIR" <<'PY'
import os, signal, sys, time
me, drzero = int(sys.argv[1]), sys.argv[2]
skip, p = set(), me
while p > 1:                                   # never ourselves or our ancestors
    skip.add(p)
    try:
        p = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        break
found = []
for d in os.listdir("/proc"):
    if not d.isdigit() or int(d) in skip:
        continue
    try:
        args = [a.decode(errors="replace") for a in open(f"/proc/{d}/cmdline", "rb").read().split(b"\0") if a]
        env = open(f"/proc/{d}/environ", "rb").read().split(b"\0")
    except OSError:
        continue
    i = next((k for k, a in enumerate(args) if a.endswith("scripts/p0/run_p0.sh")), None)
    if i is None or ("DRZERO_DIR=" + drzero).encode() not in env:
        continue
    stage = args[i + 1] if i + 1 < len(args) else ""
    if stage in ("cleanup", "probe", "check"):
        continue
    found.append((int(d), stage))
for pid, stage in found:
    print(f"[p0 cleanup] stopping run_p0.sh {stage} (pid {pid}) so it cannot start its next stage", flush=True)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
if found:
    time.sleep(1)
PY
}

finish_stage() {  # finish_stage <name> <rc>
  if [ "$2" -ne 0 ]; then log "FAILED $1 (exit $2)"; exit "$2"; fi
  log "done $1"
}

timed() {  # timed <name> <cmd...>: timed_rc, and stop the whole run on a nonzero exit
  local rc=0
  timed_rc "$@" || rc=$?
  finish_stage "$1" "$rc"
}

# Every stage's product can be checked for completeness (work3/stage_check.py): a checkpoint
# (tracker names the final step, all rank shards, data.pt, HF config), a question pool (readable
# parquet, no empty prompt / data_source / answer), a passed pool check, a converted HF model
# (config + every listed weight shard) and an evaluation dump (every test question of every
# data source). With --since, output written before the stage started never counts.
output_complete() {  # output_complete <kind> <args...>
  python -m work3.stage_check ${STAGE_T0:+--since "$STAGE_T0"} "$@"
}

settle_timing() {  # settle_timing <name>: a stage whose final checkpoint is complete must not stay "failed"
  python3 - "$OUT_ROOT/timings.txt" "$1" <<'PY'
import sys
path, name = sys.argv[1], sys.argv[2]
try:
    rows = [l.split() for l in open(path, encoding="utf-8").read().splitlines()]
except OSError:
    sys.exit(0)
rows = [r for r in rows if r[:1] == [name]]
if rows and len(rows[-1]) >= 3 and rows[-1][2] != "0":
    r = rows[-1]
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join([name, r[1], "0"] + r[3:4]) + "\n")
    print(f"[p0] {name}: its exit code {r[2]} came after a complete final checkpoint; recorded as done", flush=True)
PY
}

checked() {  # checked <name> <kind> <check args...> -- <cmd...>
  # Like timed, but a nonzero exit AFTER the stage's output was completely written (an error or a
  # SIGKILL while the process shut down; factory feedback #7, #8) is not a failure: warn, record it
  # in teardown_warnings.txt and carry on. Incomplete output still stops the run.
  local name="$1" kind="$2"; shift 2
  local cargs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do cargs+=("$1"); shift; done
  [ "${1:-}" = "--" ] || { log "checked: missing -- before the command"; exit 2; }
  shift
  local rc=0
  timed_rc "$name" "$@" || rc=$?
  if [ "$rc" -ne 0 ] && output_complete "$kind" "${cargs[@]}"; then
    log "WARNING $name exited $rc after its output was completely written ($kind check passed; an error while the process shut down, see logs/$name.log); continuing"
    echo "$name $rc $kind $(date +%s)" >> "$OUT_ROOT/teardown_warnings.txt"
    settle_timing "$name"
    rc=0
  fi
  STAGE_T0=""
  finish_stage "$name" "$rc"
}

skip_complete() {  # skip_complete <name> <kind> <check args...>: true if the output is complete
  [ "${FORCE:-0}" = "1" ] && return 1
  local name="$1"; shift
  if STAGE_T0="" output_complete "$@" > /dev/null; then
    log "skip $name: output complete ($1 check); FORCE=1 re-runs it"
    settle_timing "$name"
    return 0
  fi
  log "$name: no complete output yet, running it"
  return 1
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
import pandas, tensordict, faiss, torch, verl, numpy
print("pandas", pandas.__version__, "tensordict", tensordict.__version__, "faiss", faiss.__version__,
      "numpy", numpy.__version__)
if int(pandas.__version__.split(".")[0]) >= 3:
    bad.append("pandas >= 3 breaks upstream main_generation (copy-on-write)")
if int(numpy.__version__.split(".")[0]) >= 2:
    bad.append(f"numpy {numpy.__version__} >= 2 (pin faiss-gpu-cu12==1.8.0.2 and numpy<2)")
here = os.path.dirname(verl.__file__)
if not here.startswith(os.environ["DRZERO_DIR"]):
    bad.append(f"verl imported from {here}, not from DRZERO_DIR (uninstall pip verl)")
print("gpus", torch.cuda.device_count(), "faiss gpus", faiss.get_num_gpus())
# factory feedback #2: imports that failed on the documented install path
try:
    import compressed_tensors
    print("compressed-tensors", getattr(compressed_tensors, "__version__", "?"))
    from verl.custom_reward.reward_function import compute_challenger_format_scores  # noqa: F401
except Exception as e:
    bad.append(f"reward module import failed ({e!r}); pin compressed-tensors==0.10.2")
try:
    import flash_attn
    from flash_attn.bert_padding import pad_input  # noqa: F401  (verl dp_actor imports it on CUDA)
    print("flash-attn", flash_attn.__version__, "torch cxx11 abi", torch._C._GLIBCXX_USE_CXX11_ABI)
except Exception as e:
    bad.append(f"flash-attn import failed ({e!r}); install the wheel matching torch's cxx11 ABI")
# Reproduce the runtime situation in a fresh interpreter: sglang installs the uvloop policy on
# import, then verl's fsdp_sglang.py calls get_event_loop() in a main thread that never set a loop.
# (r2-r6 also called set_event_loop(None) first, which fails on every uvloop version -- removed
# after factory feedback #5.)
import subprocess
probe = subprocess.run([sys.executable, "-c",
    "import asyncio, uvloop; asyncio.set_event_loop_policy(uvloop.EventLoopPolicy()); "
    "asyncio.get_event_loop(); print(uvloop.__version__)"], capture_output=True, text=True)
if probe.returncode == 0:
    print("uvloop", probe.stdout.strip(), "get_event_loop ok")
else:
    bad.append("uvloop get_event_loop failed (" + (probe.stderr.strip().splitlines() or ["?"])[-1]
               + "); pin uvloop==0.21.0")
if bad:
    print("PROBLEMS:", *bad, sep="\n  ")
    sys.exit(2)
PY
  [ "$bad" = 0 ] || { log "check FAILED"; exit 2; }
  log "check ok"
}

stage_prepare() {
  [ -z "$ARM" ] || { log "prepare is P0-only; P2 arms reuse P0's data (scripts/p2/run_p2.sh setup)"; exit 2; }
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
        -e "s#^CONFIG_PATH=\"./config\"\$#CONFIG_PATH=\"$DRZERO_DIR/config\"#"
      )
      # Hydra resolves a relative --config-path against verl/trainer/main_ppo.py, not the cwd
      # (factory feedback #2, bug 3), hence the absolute CONFIG_PATH above.
      if [ "$kind" = challenger ]; then
        sedargs+=(-e "s#actor_rollout_ref.rollout.gpu_memory_utilization=\${rollout_memory_utilization} \\\\\$#actor_rollout_ref.rollout.gpu_memory_utilization=$CHAL_ROLLOUT_MEM \\\\#")
        # reward server layout is set explicitly; the in-trainer `tp` variable below no longer feeds it
        sedargs+=(-e "s#^    --tp-size=\${tp} \\\\\$#    --tp-size=$REWARD_TP \\\\#"
                  -e "s#^    --dp-size=\${dp} \\\\\$#    --dp-size=$REWARD_DP \\\\#")
        [ "$REWARD_METRICS" = "1" ] && sedargs+=(-e "s#^    --log-level=error &\$#    --log-level=error --enable-metrics \\&#")
      fi
      if [ "$kind" != gen_data ]; then
        sedargs+=(-e "s#^tp=2\$#tp=$ROLLOUT_TP#")
        # actor update / log-prob micro-batch (mini-batch 256 and the optimiser step are unchanged)
        sedargs+=(-e "s#^batch_per_gpu=2\$#batch_per_gpu=$TRAIN_MICRO#")
        # crash recovery: checkpoint more often (keep only the latest few) so a crash loses at most
        # SAVE_FREQ steps; verl's default resume_mode=auto picks the latest checkpoint up again
        sedargs+=(-e "s#^    trainer.save_freq=25 \\\\\$#    trainer.save_freq=$SAVE_FREQ \\\\\\n    trainer.max_actor_ckpt_to_keep=$CKPT_KEEP \\\\#")
        # proposer iterations 2-3 force resume_path (previous iteration); run_p0.sh switches this
        # to auto when the current iteration already has its own checkpoints
        sedargs+=(-e "s#^    trainer.resume_mode=\"resume_path\" \\\\\$#    trainer.resume_mode=\"\${WORK3_RESUME_MODE:-resume_path}\" \\\\#")
      fi
      [ "$FAISS_GPU" = "1" ] || sedargs+=(-e "/^ *--faiss_gpu \\\\\$/d")
      [ -z "${GEN_MEM:-}" ] || sedargs+=(-e "s#^rollout_memory_utilization=0.8\$#rollout_memory_utilization=$GEN_MEM#")
      if [ -n "$ARM" ] && [ "$kind" = challenger ]; then
        sedargs+=(
          -e "s#custom_reward_function.name=compute_challenger_score_batch \\\\\$#custom_reward_function.name=compute_challenger_score_batch_arm \\\\#"
          -e "s#custom_reward_function.path=verl/custom_reward/reward_function.py \\\\\$#custom_reward_function.path=$WORK3_DIR/work3/ncp_reward.py \\\\\\n    +custom_reward_function.reward_kwargs.arm=$ARM \\\\#"
        )
      fi
      # write to a temp file and rename: a stage that is currently executing an older patched
      # script keeps reading its own (unlinked) copy, so re-patching mid-run is safe
      sed "${sedargs[@]}" "iter${t}_${kind}.sh" > "$PATCHED/.iter${t}_${kind}.sh.tmp"
      mv -f "$PATCHED/.iter${t}_${kind}.sh.tmp" "$PATCHED/iter${t}_${kind}.sh"
    done
  done
  if [ -n "$ARM" ]; then
    local n_arm; n_arm=$(grep -l "reward_kwargs.arm=$ARM" "$PATCHED"/iter*_challenger.sh | wc -l)
    [ "$n_arm" = 3 ] || { log "patch FAILED: reward arm inserted in $n_arm of 3 proposer scripts"; exit 2; }
    if grep -n "verl/custom_reward/reward_function.py" "$PATCHED"/iter*_challenger.sh; then
      log "patch FAILED: upstream reward still referenced"; exit 2
    fi
    log "proposer reward arm: $ARM"
  fi
  # every upstream port reference must be gone
  if grep -n "800[01]\|lsof" "$PATCHED"/iter*.sh "$PATCHED/search_tool_config.yaml"; then
    log "patch FAILED: upstream port references left"; exit 2
  fi
  local n_wait; n_wait=$(grep -l "wait_retriever.sh" "$PATCHED"/iter*.sh | wc -l)
  [ "$n_wait" = 9 ] || { log "patch FAILED: retriever wait inserted in $n_wait of 9 scripts"; exit 2; }
  local n_cfg; n_cfg=$(grep -l "^CONFIG_PATH=\"$DRZERO_DIR/config\"\$" "$PATCHED"/iter*.sh | wc -l)
  [ "$n_cfg" = 9 ] || { log "patch FAILED: absolute CONFIG_PATH in $n_cfg of 9 scripts"; exit 2; }
  local n_mem; n_mem=$(grep -l "rollout.gpu_memory_utilization=$CHAL_ROLLOUT_MEM " "$PATCHED"/iter*_challenger.sh | wc -l)
  [ "$n_mem" = 3 ] || { log "patch FAILED: proposer rollout memory set in $n_mem of 3 scripts"; exit 2; }
  local n_rl; n_rl=$(grep -l "^    --tp-size=$REWARD_TP \\\\\$" "$PATCHED"/iter*_challenger.sh | wc -l)
  [ "$n_rl" = 3 ] || { log "patch FAILED: reward server layout set in $n_rl of 3 scripts"; exit 2; }
  local n_tp; n_tp=$(grep -l "^tp=$ROLLOUT_TP\$" "$PATCHED"/iter*_challenger.sh "$PATCHED"/iter*_solver.sh | wc -l)
  [ "$n_tp" = 6 ] || { log "patch FAILED: in-trainer tp set in $n_tp of 6 scripts"; exit 2; }
  if grep -n -- "--tp-size=\${tp}\|--dp-size=\${dp}" "$PATCHED"/iter*_challenger.sh; then
    log "patch FAILED: reward server still follows the in-trainer tp/dp"; exit 2
  fi
  local n_sv; n_sv=$(grep -l "trainer.max_actor_ckpt_to_keep=$CKPT_KEEP " "$PATCHED"/iter*_challenger.sh "$PATCHED"/iter*_solver.sh | wc -l)
  [ "$n_sv" = 6 ] || { log "patch FAILED: checkpoint frequency set in $n_sv of 6 scripts"; exit 2; }
  local n_rm; n_rm=$(grep -l 'WORK3_RESUME_MODE' "$PATCHED"/iter2_challenger.sh "$PATCHED"/iter3_challenger.sh | wc -l)
  [ "$n_rm" = 2 ] || { log "patch FAILED: resume switch in $n_rm of 2 proposer scripts"; exit 2; }
  local n_mb; n_mb=$(grep -l "^batch_per_gpu=$TRAIN_MICRO\$" "$PATCHED"/iter*_challenger.sh "$PATCHED"/iter*_solver.sh | wc -l)
  [ "$n_mb" = 6 ] || { log "patch FAILED: micro-batch set in $n_mb of 6 scripts"; exit 2; }
  log "checkpoints: every $SAVE_FREQ steps, keep $CKPT_KEEP; proposer rollout memory $CHAL_ROLLOUT_MEM; micro-batch $TRAIN_MICRO${PYTORCH_CUDA_ALLOC_CONF:+; $PYTORCH_CUDA_ALLOC_CONF}"
  log "parallel layout: in-trainer rollout tp=$ROLLOUT_TP; reward server tp=$REWARD_TP dp=$REWARD_DP (metrics=$REWARD_METRICS)"
  grep -n "^gpus=\|^dp=\|^model=\|conda activate\|faiss_gpu\|retriever_model\|^rollout_memory\|gpu_memory_utilization\|mem-fraction\|p0_launch\|wait_retriever\|CONFIG_PATH=\|TOOL_CONFIG=\|--port=\|base_url" "$PATCHED"/iter1_*.sh
  log "patched scripts in $PATCHED"
}

convert_solver() {  # convert_solver <t>
  local base="$CKPT/solver_iter$1_ratio${RATIO}_grpo_group5_${MODEL_NAME}"
  need "$base/global_step_50/actor"
  python -m verl.model_merger merge --backend fsdp \
    --local_dir "$base/global_step_50/actor" --target_dir "$base/solver_iter$1_hf"
}


stage_iter() {  # stage_iter <t>; re-running it after a crash resumes where it stopped
  local t="$1"
  local chal="$CKPT/challenger_iter${t}_ratio${RATIO}_grpo_batch_group1-5_${MODEL_NAME}"
  local solv="$CKPT/solver_iter${t}_ratio${RATIO}_grpo_group5_${MODEL_NAME}"
  local pool="data/zero_challenger_iter${t}_ratio${RATIO}_grpo_batch_group1-5_${MODEL_NAME}.parquet"
  need "$PATCHED/iter${t}_challenger.sh"; need data/test_1200.parquet
  if [ "$t" -gt 1 ]; then
    need "$CKPT/solver_iter$((t-1))_ratio${RATIO}_grpo_group5_${MODEL_NAME}/solver_iter$((t-1))_hf/config.json"
  fi
  cleanup_servers
  if ! skip_complete "iter${t}_challenger" ckpt "$chal" "$((50 * t))" "$NGPU"; then
    # iterations 2-3 normally start from the previous iteration's checkpoint; if this iteration
    # already saved its own (a crash), continue from the latest one instead
    if [ "$t" -gt 1 ] && ls -d "$chal"/global_step_* >/dev/null 2>&1; then
      export WORK3_RESUME_MODE=auto
      log "iter${t}_challenger: resuming from its own latest checkpoint"
    fi
    checked "iter${t}_challenger" ckpt "$chal" "$((50 * t))" "$NGPU" -- \
      bash "$PATCHED/iter${t}_challenger.sh" "$RATIO" "trainer.total_training_steps=$((50 * t))"
    unset WORK3_RESUME_MODE
  fi
  skip_complete "iter${t}_gen_data" pool "$pool" ||
    checked "iter${t}_gen_data" pool "$pool" -- bash "$PATCHED/iter${t}_gen_data.sh" "$RATIO"
  checked "iter${t}_pool_check" pool_check "$OUT_ROOT/pool_check_iter${t}.json" -- \
    python -m work3.pool_check "$pool" --out "$OUT_ROOT/pool_check_iter${t}.json"
  skip_complete "iter${t}_solver" ckpt "$solv" 50 "$NGPU" ||
    checked "iter${t}_solver" ckpt "$solv" 50 "$NGPU" -- \
      bash "$PATCHED/iter${t}_solver.sh" "$RATIO" "trainer.total_training_steps=50"
  skip_complete "iter${t}_convert" hf "$solv/solver_iter${t}_hf" ||
    checked "iter${t}_convert" hf "$solv/solver_iter${t}_hf" -- convert_solver "$t"
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
    --config-path="$DRZERO_DIR/config" --config-name=search_multiturn_grpo \
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
    actor_rollout_ref.rollout.gpu_memory_utilization="${EVAL_ROLLOUT_MEM:-0.6}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
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
  skip_complete "eval${t}" eval "$OUT_ROOT/eval/iter$t" data/test.parquet && return 0
  checked "eval${t}" eval "$OUT_ROOT/eval/iter$t" data/test.parquet -- eval_model "$t" "$model"
}

stage_all() {  # unattended: every remaining P0 stage in order; finished stages are skipped
  local s attempt max="${MAX_ATTEMPTS:-2}"
  for s in ${ALL_STAGES:-iter1 eval1 iter2 eval2 iter3 eval3 eval0 gate}; do
    attempt=1
    until WORK3_P0_LOCK_HELD=1 bash "$WORK3_DIR/scripts/p0/run_p0.sh" "$s"; do
      if [ "$attempt" -ge "$max" ]; then
        log "all: $s failed $attempt time(s); stopping (logs in $OUT_ROOT/logs)"
        exit 1
      fi
      attempt=$((attempt + 1))
      log "all: $s failed; cleaning up and retrying from where it stopped (attempt $attempt of $max)"
      cleanup_servers
      sleep 60
    done
  done
  log "all: P0 finished; summary $OUT_ROOT/p0_summary.json"
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
  prepare|patch|iter1|iter2|iter3|eval0|eval1|eval2|eval3|gate|all) take_lock ;;
esac

case "${1:-}" in
  check) stage_check ;;
  prepare) stage_prepare ;;
  patch) stage_patch ;;
  iter1|iter2|iter3) stage_iter "${1#iter}" ;;
  eval0|eval1|eval2|eval3) stage_eval "${1#eval}" ;;
  gate) stage_gate ;;
  all) stage_all ;;
  cleanup) stop_orchestrators; cleanup_servers ;;
  probe)   # read-only speed diagnosis of whatever stage is running now (factory feedback #3)
    python -m work3.speed_probe --seconds "${PROBE_SECONDS:-120}" --reward-port "$REWARD_PORT" \
      --retriever-stats "$OUT_ROOT/logs/retriever_stats.json" \
      --out "$OUT_ROOT/speed_probe_$(date +%Y%m%d_%H%M%S).json" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
