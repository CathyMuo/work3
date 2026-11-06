#!/usr/bin/env bash
# P2: proposer-reward arms (beta, ncp) against the P0 Dr. Zero run.
# Protocol: research/protocols/P2_ncp_pilot.md (frozen). Run only after P1 is GO.
#
# Usage:
#   bash scripts/p2/run_p2.sh setup <arm>            # arm = beta | ncp
#   bash scripts/p2/run_p2.sh <arm> <p0-stage>       # check|patch|iter1|iter2|iter3|eval1|eval2|eval3|gate|cleanup
#   bash scripts/p2/run_p2.sh p1 <arm>               # P1 diagnosis on the arm's pool 3 (secondary)
#   bash scripts/p2/run_p2.sh gate                   # writes $ARM_ROOT/p2_summary.json
#
# Required env: P0_DIR (the P0 Dr. Zero checkout = P0's DRZERO_DIR), MODEL_3B, plus the
#   P0 variables (E5_MODEL, NGPU, ports ...). Optional: ARM_ROOT ($(dirname $P0_DIR)/drzero_p2)
set -euo pipefail

PIN=98c4a6dd2495e85d8b4d947ab4c030e540c922c5
WORK3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${P0_DIR:?set P0_DIR to the P0 DRZERO_DIR}"
ARM_ROOT="${ARM_ROOT:-$(dirname "$P0_DIR")/drzero_p2}"
log() { echo "[p2 $(date '+%F %T')] $*"; }
arm_ok() { case "${1:-}" in beta|ncp) ;; *) log "arm must be beta or ncp"; exit 2 ;; esac; }

setup_arm() {
  local arm="$1" dir="$ARM_ROOT/$1"
  arm_ok "$arm"
  for f in data/zero_ratio4321.parquet data/test.parquet data/test_1200.parquet corpus/wiki-18.jsonl corpus/e5_Flat.index; do
    [ -e "$P0_DIR/$f" ] || { log "missing P0 file: $P0_DIR/$f"; exit 2; }
  done
  mkdir -p "$ARM_ROOT"
  if [ ! -d "$dir" ]; then
    git -C "$P0_DIR" worktree add --detach "$dir" "$PIN"
  fi
  [ "$(git -C "$dir" rev-parse HEAD)" = "$PIN" ] || { log "$dir is not at $PIN"; exit 2; }
  mkdir -p "$dir/data" "$dir/corpus"
  # identical proposer seed prompts (hop draws are unseeded upstream) and evaluation files
  for f in zero_ratio4321.parquet test.parquet test_1200.parquet; do
    ln -sfn "$(readlink -f "$P0_DIR/data/$f")" "$dir/data/$f"
  done
  for f in wiki-18.jsonl e5_Flat.index; do
    ln -sfn "$(readlink -f "$P0_DIR/corpus/$f")" "$dir/corpus/$f"
  done
  ls -l "$dir/data" "$dir/corpus"
  log "arm $arm ready in $dir"
}

run_arm() {  # run_arm <arm> <stage>
  local arm="$1" stage="$2"
  arm_ok "$arm"
  [ "$stage" != prepare ] || { log "prepare is P0-only"; exit 2; }
  [ -d "$ARM_ROOT/$arm" ] || { log "run: bash $0 setup $arm"; exit 2; }
  DRZERO_DIR="$ARM_ROOT/$arm" P0_REWARD_ARM="$arm" bash "$WORK3_DIR/scripts/p0/run_p0.sh" "$stage"
}

run_p1_arm() {
  local arm="$1" dir="$ARM_ROOT/$1"
  arm_ok "$arm"
  DRZERO_DIR="$dir" P0_OUT="$dir/work3_p0" bash "$WORK3_DIR/scripts/p1/run_p1.sh" pool3
  DRZERO_DIR="$dir" P0_OUT="$dir/work3_p0" bash "$WORK3_DIR/scripts/p1/run_p1.sh" summary || true
}

gate() {
  local args=(--drzero "$P0_DIR/work3_p0/p0_summary.json"
              --beta "$ARM_ROOT/beta/work3_p0/p0_summary.json"
              --ncp "$ARM_ROOT/ncp/work3_p0/p0_summary.json")
  for a in beta ncp; do
    [ -f "$ARM_ROOT/$a/work3_p0/reward_log.jsonl" ] && args+=("--${a}-reward-log" "$ARM_ROOT/$a/work3_p0/reward_log.jsonl")
    [ -f "$ARM_ROOT/$a/work3_p1/p1_summary.json" ] && args+=("--${a}-p1" "$ARM_ROOT/$a/work3_p1/p1_summary.json")
  done
  [ -f "$P0_DIR/work3_p1/p1_summary.json" ] && args+=(--drzero-p1 "$P0_DIR/work3_p1/p1_summary.json")
  PYTHONPATH="$WORK3_DIR:${PYTHONPATH:-}" python -m work3.p2_gate "${args[@]}" --out "$ARM_ROOT/p2_summary.json" || true
  log "summary: $ARM_ROOT/p2_summary.json"
}

case "${1:-}" in
  setup) setup_arm "${2:-}" ;;
  beta|ncp) run_arm "$1" "${2:?stage}" ;;
  p1) run_p1_arm "${2:-}" ;;
  gate) gate ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
