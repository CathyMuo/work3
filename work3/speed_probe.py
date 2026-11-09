"""Read-only speed diagnosis for a running P0 / P2 stage (factory feedback #3).

    python -m work3.speed_probe [--seconds 120] [--interval 2] [--reward-port 8101]
        [--retriever-stats $DRZERO_DIR/work3_p0/logs/retriever_stats.json] [--out speed_probe.json]

Samples, without touching the run:
  * per-GPU utilisation / memory / power (nvidia-smi) and which processes sit on which GPU;
  * the proposer-reward SGLang server: parallel layout and KV capacity (/get_server_info) and,
    if started with --enable-metrics, running / queued requests, KV usage and throughput (/metrics);
  * the retriever latency file written by work3.p0_launch.
Prints a short Chinese verdict and writes the raw samples as JSON. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_sample() -> List[dict]:
    out = []
    for line in _run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
                      "--format=csv,noheader,nounits"]).strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 5:
            num = lambda x: float(x) if re.fullmatch(r"[\d.]+", x) else None
            out.append({"index": p[0], "uuid": p[1], "util": num(p[2]), "mem": num(p[3]), "power": num(p[4])})
    return out


def gpu_processes(uuid_to_index: Dict[str, str]) -> List[dict]:
    rows = []
    for line in _run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                      "--format=csv,noheader,nounits"]).strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) != 3:
            continue
        try:
            cmd = Path(f"/proc/{p[0]}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            cmd = "?"
        kind = ("reward-sglang" if "sglang.launch_server" in cmd else
                "retriever" if "retrieval_server" in cmd else
                "ray-worker" if "ray::" in cmd or "default_worker" in cmd else cmd[:60])
        rows.append({"pid": int(p[0]), "gpu": uuid_to_index.get(p[1], p[1]), "mem_mb": float(p[2] or 0),
                     "kind": kind})
    return rows


def http_get(url: str, timeout: float = 5.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - absent server or endpoint
        return None


def parse_prometheus(text: str) -> Dict[str, List[float]]:
    """name -> values over all label sets (e.g. one per DP / TP rank)."""
    out: Dict[str, List[float]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+|NaN)", line)
        if m:
            try:
                out.setdefault(m.group(1), []).append(float(m.group(3)))
            except ValueError:
                pass
    return out


def reward_metrics(port: int) -> Optional[dict]:
    text = http_get(f"http://127.0.0.1:{port}/metrics")
    if not text or "sglang:" not in text:
        return None
    m = parse_prometheus(text)
    pick = lambda k, f: f(m[k]) if m.get(k) else None
    return {"running": pick("sglang:num_running_reqs", sum), "queued": pick("sglang:num_queue_reqs", sum),
            "token_usage_max": pick("sglang:token_usage", max), "gen_throughput": pick("sglang:gen_throughput", sum),
            "ranks": len(m.get("sglang:num_running_reqs", []))}


def server_info(port: int) -> Optional[dict]:
    text = http_get(f"http://127.0.0.1:{port}/get_server_info")
    if not text:
        return None
    try:
        d = json.loads(text)
    except ValueError:
        return None
    keep = ("tp_size", "dp_size", "mem_fraction_static", "max_total_num_tokens", "max_running_requests",
            "enable_metrics", "base_gpu_id", "load_balance_method", "version", "model_path")
    return {k: d.get(k) for k in keep if k in d}


def summarize(samples: List[dict], info: Optional[dict], procs: List[dict], retr: Optional[dict]) -> dict:
    gpus: Dict[str, List[dict]] = {}
    for s in samples:
        for g in s["gpus"]:
            gpus.setdefault(g["index"], []).append(g)
    per_gpu = {}
    for idx, rows in sorted(gpus.items()):
        utils = [r["util"] for r in rows if r["util"] is not None]
        per_gpu[idx] = {"util_mean": round(sum(utils) / len(utils), 1) if utils else None,
                        "busy_share": round(sum(u > 0 for u in utils) / len(utils), 2) if utils else None,
                        "power_mean": round(sum(r["power"] or 0 for r in rows) / len(rows), 1),
                        "mem_max": max((r["mem"] or 0) for r in rows),
                        "processes": sorted({p["kind"] for p in procs if p["gpu"] == idx})}
    rm = [s["reward"] for s in samples if s.get("reward")]
    reward = None
    if rm:
        avg = lambda k: round(sum(x[k] or 0 for x in rm) / len(rm), 2)
        reward = {"running_mean": avg("running"), "queued_mean": avg("queued"),
                  "queued_share": round(sum((x["queued"] or 0) > 0 for x in rm) / len(rm), 2),
                  "token_usage_max": max(x["token_usage_max"] or 0 for x in rm),
                  "gen_throughput_mean": avg("gen_throughput"), "ranks": rm[-1]["ranks"]}
    verdict = []
    idle = [i for i, g in per_gpu.items() if g["busy_share"] == 0]
    if idle:
        verdict.append(f"GPU {','.join(idle)} 在整个采样期间利用率都是 0。"
                       + ("其上没有奖励服务进程。" if all("reward-sglang" not in per_gpu[i]["processes"] for i in idle)
                          else "其上有奖励服务进程却没有计算，奖励服务的 DP 分发可能没有生效。"))
    if info:
        verdict.append(f"奖励服务：tp={info.get('tp_size')}，dp={info.get('dp_size')}，显存比例 "
                       f"{info.get('mem_fraction_static')}，KV 容量 {info.get('max_total_num_tokens')} token。")
    if reward:
        if reward["token_usage_max"] >= 0.95 and reward["queued_share"] >= 0.3:
            verdict.append("奖励服务 KV 缓存经常占满且有请求排队：吞吐受显存限制。")
        elif reward["running_mean"] < 32 and reward["queued_mean"] == 0:
            verdict.append("奖励服务同时只在处理很少的请求、也没有排队：时间花在等待（长尾的多轮生成或检索），不是算力或显存不足。")
        else:
            verdict.append("奖励服务负载正常，没有明显排队。")
    else:
        verdict.append("奖励服务没有开启 /metrics（p0-r5 起的 patch 会开启），无法判断排队与 KV 占用。")
    if retr:
        verdict.append(f"检索：p50 {retr.get('p50_ms')} ms，p95 {retr.get('p95_ms')} ms，进行中 {retr.get('in_flight')}。"
                       + ("检索偏慢，可能是瓶颈。" if (retr.get("p95_ms") or 0) > 1000 else ""))
    return {"per_gpu": per_gpu, "reward_server": info, "reward_load": reward, "retriever": retr,
            "processes": procs, "verdict": verdict}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--reward-port", type=int, default=int(os.environ.get("REWARD_PORT", 8101)))
    ap.add_argument("--retriever-stats")
    ap.add_argument("--out", default="speed_probe.json")
    a = ap.parse_args(argv)
    first = gpu_sample()
    procs = gpu_processes({g["uuid"]: g["index"] for g in first})
    info = server_info(a.reward_port)
    samples, t_end = [], time.time() + a.seconds
    while True:
        samples.append({"t": time.time(), "gpus": gpu_sample(), "reward": reward_metrics(a.reward_port)})
        if time.time() >= t_end:
            break
        time.sleep(a.interval)
    retr = None
    if a.retriever_stats and Path(a.retriever_stats).exists():
        try:
            retr = json.loads(Path(a.retriever_stats).read_text(encoding="utf-8"))
        except ValueError:
            retr = None
    res = summarize(samples, info, procs, retr)
    Path(a.out).write_text(json.dumps({"summary": res, "samples": samples}, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("per_gpu", "reward_server", "reward_load", "retriever")},
                     ensure_ascii=False, indent=1))
    print("\n".join("· " + v for v in res["verdict"]))
    print(f"(原始采样已写入 {a.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
