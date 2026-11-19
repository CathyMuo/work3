#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/run_wiki_rsi_w2a.sh overnight-prepare "$@"
exec bash scripts/run_wiki_rsi_w2a.sh overnight "$@"
