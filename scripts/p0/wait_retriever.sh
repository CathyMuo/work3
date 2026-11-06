#!/usr/bin/env bash
# Block until the retriever on port $1 answers a query (max ${2:-1800} s).
# Inserted by run_p0.sh patch after each upstream retriever launch, because the
# upstream scripts only `sleep 30` and the e5 Flat index can take minutes to load;
# searches sent before that fail silently inside training rollouts.
port="${1:?port}"; limit="${2:-1800}"; t0=$(date +%s)
while :; do
  if curl -s -o /dev/null -f -X POST "http://127.0.0.1:$port/retrieve" \
       -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}'; then
    echo "[p0] retriever ready on :$port after $(( $(date +%s) - t0 )) s"
    exit 0
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$limit" ]; then
    echo "[p0] retriever on :$port not ready after $limit s" >&2
    exit 3
  fi
  sleep 10
done
