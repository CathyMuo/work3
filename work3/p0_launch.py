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


def install_port_override(port: int) -> None:
    import uvicorn

    original = uvicorn.run

    def run(app, *args, **kwargs):
        kwargs["port"] = port
        print(f"[p0_launch] uvicorn port forced to {port}", flush=True)
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
