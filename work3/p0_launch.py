"""Run pinned upstream scripts unchanged, with two launch-time substitutions.

    python -m work3.p0_launch offline-hf --map train.parquet=/abs/train.parquet \
        [--map test.parquet=/abs/test.parquet] -- SCRIPT.py [args...]
        Replace huggingface_hub.hf_hub_download with a lookup of local files by
        filename (Dr. Zero's process_train.py / process_test.py always download).

    python -m work3.p0_launch serve --port 8100 -- search/retrieval_server.py [args...]
        Force the port passed to uvicorn.run (the upstream retriever hard-codes 8000).

The target script runs via runpy as __main__ with its own argv, so upstream files
stay byte-identical to the pinned commit.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys


def parse_map(items: list[str]) -> dict[str, str]:
    mapping = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--map expects FILENAME=PATH, got {item!r}")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        mapping[name] = os.path.abspath(path)
    return mapping


def make_offline_download(mapping: dict[str, str]):
    def hf_hub_download(repo_id=None, filename=None, *args, **kwargs):
        if filename not in mapping:
            raise FileNotFoundError(
                f"offline mode: no local file mapped for {repo_id}/{filename}")
        print(f"[p0_launch] offline: {repo_id}/{filename} -> {mapping[filename]}", flush=True)
        return mapping[filename]
    return hf_hub_download


def install_offline_hf(mapping: dict[str, str]) -> None:
    import huggingface_hub

    huggingface_hub.hf_hub_download = make_offline_download(mapping)
    try:
        import huggingface_hub.file_download as fd
        fd.hf_hub_download = huggingface_hub.hf_hub_download
    except ImportError:
        pass


class LatencyStats:
    """ASGI wrapper: per-request latency of the retriever, summarised to a JSON file.

    Read-only observability for the board (is rollout time spent waiting on search?).
    Writes $WORK3_RETRIEVER_STATS at most every `every` seconds; never alters responses.
    """

    def __init__(self, app, path: str | None, every: float = 15.0, window: float = 300.0):
        import collections
        import threading

        self.app, self.path, self.every, self.window = app, path, every, window
        self.events = collections.deque()          # (end_time, seconds, status)
        self.in_flight = 0
        self.total = 0
        self.errors = 0
        self.last_write = 0.0
        self.lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        import time

        status = {"code": 0}

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 0))
            await send(message)

        t0 = time.perf_counter()
        with self.lock:
            self.in_flight += 1
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.record(time.time(), time.perf_counter() - t0, status["code"])

    def record(self, now: float, seconds: float, code: int) -> None:
        with self.lock:
            self.in_flight -= 1
            self.total += 1
            self.errors += int(code >= 500 or code == 0)
            self.events.append((now, seconds, code))
            while self.events and now - self.events[0][0] > self.window:
                self.events.popleft()
            due = self.path and now - self.last_write >= self.every
            if due:
                self.last_write = now
                snap = self.snapshot(now)
        if due:
            self.write(snap)

    def snapshot(self, now: float) -> dict:
        lat = sorted(s for _, s, _ in self.events)
        q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))] * 1000, 1) if lat else None
        span = min(self.window, max(1.0, now - self.events[0][0])) if self.events else self.window
        return {"updated": now, "window_seconds": self.window, "requests_in_window": len(lat),
                "requests_per_second": round(len(lat) / span, 3),
                "errors_in_window": sum(1 for _, _, c in self.events if c >= 500 or c == 0),
                "p50_ms": q(0.5), "p95_ms": q(0.95), "max_ms": q(1.0),
                "mean_ms": round(sum(lat) / len(lat) * 1000, 1) if lat else None,
                "in_flight": self.in_flight, "total_requests": self.total, "total_errors": self.errors}

    def write(self, snap: dict) -> None:
        import json as _json

        tmp = f"{self.path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(snap, f)
            os.replace(tmp, self.path)
        except OSError:
            pass


def install_port_override(port: int) -> None:
    import uvicorn

    original = uvicorn.run

    def run(app, *args, **kwargs):
        kwargs["port"] = port
        print(f"[p0_launch] uvicorn port forced to {port}", flush=True)
        stats_path = os.environ.get("WORK3_RETRIEVER_STATS")
        if stats_path:
            app = LatencyStats(app, stats_path)
            print(f"[p0_launch] retriever latency stats -> {stats_path}", flush=True)
        return original(app, *args, **kwargs)

    uvicorn.run = run


def run_script(script: str, args: list[str]) -> None:
    sys.argv = [script, *args]
    script_dir = os.path.dirname(os.path.abspath(script))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    runpy.run_path(script, run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m work3.p0_launch {offline-hf|serve} ... -- SCRIPT [args]")
    split = argv.index("--")
    own, target = argv[:split], argv[split + 1:]
    if not target:
        raise SystemExit("missing target script after --")
    ap = argparse.ArgumentParser(prog="work3.p0_launch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    off = sub.add_parser("offline-hf")
    off.add_argument("--map", action="append", default=[], required=True)
    srv = sub.add_parser("serve")
    srv.add_argument("--port", type=int, required=True)
    ns = ap.parse_args(own)
    if ns.cmd == "offline-hf":
        install_offline_hf(parse_map(ns.map))
    else:
        install_port_override(ns.port)
    run_script(target[0], target[1:])


if __name__ == "__main__":
    main()
