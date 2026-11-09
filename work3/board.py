"""Work 3 progress board: read-only local web page (research briefing + live run progress).

    bash scripts/run_board.sh            # http://127.0.0.1:8766/
    python -m work3.board [--host 127.0.0.1] [--port 8766] [--drzero-dir DIR] [--arm-root DIR]

Standard library only. It never writes anything; it only reads
  research/EXPERIMENT_PLAN.md, research/STATE.json, research/DECISION_LOG.md (this repo)
  <run>/timings.txt, <run>/logs/*.log (tails), <run>/*.json summaries, <run>/eval/iter*/ dumps
  nvidia-smi (optional, 5 s timeout)
Run directories: $DRZERO_DIR/work3_p0, $DRZERO_DIR/work3_p1 and, for P2,
$ARM_ROOT/{beta,ncp}/work3_p0|work3_p1 (env BOARD_P0_OUT / BOARD_P1_OUT override).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).resolve().parent / "board_page.html"
STALL_SECONDS = 20 * 60
TAIL_BYTES = 400_000
P0_THRESHOLD = 0.306

# Stage plan per run kind: (stage name, display name, default hours for the ETA)
P0_STAGES = [("prepare_train", "准备训练种子", 0.3), ("prepare_test", "准备测试集", 0.1),
             ("prepare_val", "抽取验证集", 0.02)]
for _t in (1, 2, 3):
    P0_STAGES += [(f"iter{_t}_challenger", f"第{_t}轮 出题者训练（50 步）", 8.0),
                  (f"iter{_t}_gen_data", f"第{_t}轮 生成题池", 4.0),
                  (f"iter{_t}_pool_check", f"第{_t}轮 题池检查", 0.02),
                  (f"iter{_t}_solver", f"第{_t}轮 解题者训练（50 步）", 6.0),
                  (f"iter{_t}_convert", f"第{_t}轮 转 HF", 0.1),
                  (f"eval{_t}", f"第{_t}轮 评测（51,713 题）", 3.0)]
P0_STAGES.append(("eval0", "基座模型评测（参考）", 3.0))
ARM_STAGES = [s for s in P0_STAGES if not s[0].startswith("prepare") and s[0] != "eval0"]
P1_STAGES = [("pool1", "题池 1 诊断（基座解题）", 2.0), ("pool3", "题池 3 诊断（solver_iter2）", 2.0),
             ("parity", "与上游提示一致性检查", 0.5)]
P1_ARM_STAGES = [("pool3", "题池 3 诊断", 2.0)]

ERROR_PATTERNS = {"traceback": re.compile(r"Traceback \(most recent call last\)"),
                  "failed": re.compile(r"\bFAILED\b"),
                  "oom": re.compile(r"CUDA out of memory|OutOfMemoryError"),
                  "undefined_function": re.compile(r"undefined function: (\w+)")}
TQDM = re.compile(r"(?:(?P<desc>[^\r\n|:]{1,60}):\s*)?(?P<pct>\d{1,3})%\|[^|\r\n]*\|\s*(?P<n>\d+)/(?P<total>\d+)"
                  r"(?:\s*\[(?P<info>[^\]\r\n]*)\])?")
STEP_LINE = re.compile(r"step:(\d+) - (.*)")
FINAL_REWARD = re.compile(r"Final rewards: Avg ([-\d.eE]+)")
ARM_REWARD = re.compile(r"\[work3 reward (\w+)\] call (\d+): .*mean difficulty ([-\d.]+)")


# ---------------------------------------------------------------- small helpers
def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def tail_text(path: Path, nbytes: int = TAIL_BYTES) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def clean_lines(text: str) -> List[str]:
    """Split on newlines and carriage returns (tqdm), drop ANSI codes and empty lines."""
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    return [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]


def parse_timings(path: Path) -> Dict[str, dict]:
    """Last entry per stage. Lines: `name seconds [exit_code end_epoch]`."""
    out: Dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        rc = int(parts[2]) if len(parts) >= 3 and parts[2].lstrip("-").isdigit() else None
        end = float(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else None
        out[parts[0]] = {"seconds": int(parts[1]), "rc": rc, "end": end}
    return out


def tqdm_progress(text: str) -> Dict[str, dict]:
    """Latest progress per bar description."""
    bars: Dict[str, dict] = {}
    for m in TQDM.finditer(text):
        desc = (m.group("desc") or "").strip() or "progress"
        bars[desc] = {"n": int(m.group("n")), "total": int(m.group("total")),
                      "pct": int(m.group("pct")), "info": (m.group("info") or "").strip()}
    return bars


def step_metrics(text: str, keys=("critic/score/mean", "response_length/mean", "timing_s/step",
                                  "actor/entropy")) -> List[dict]:
    rows = []
    for line in clean_lines(text):
        m = STEP_LINE.search(line)
        if not m:
            continue
        row = {"step": int(m.group(1))}
        for item in m.group(2).split(" - "):
            k, _, v = item.partition(":")
            if k in keys:
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        rows.append(row)
    return rows


def reward_series(text: str) -> List[float]:
    vals = [float(x) for x in FINAL_REWARD.findall(text)]
    if not vals:
        vals = [float(m[2]) for m in ARM_REWARD.findall(text)]
    return vals


def _num(v: str) -> Optional[float]:
    try:
        x = float(v)
    except ValueError:
        return None
    return x if x == x and abs(x) != float("inf") else None    # NaN / inf are not JSON


def parse_step_items(body: str) -> Dict[str, float]:
    """All numeric `key:value` items of one verl console metrics line."""
    out = {}
    for item in body.split(" - "):
        k, sep, v = item.partition(":")
        if sep and k and " " not in k.strip():
            x = _num(v.strip())
            if x is not None:
                out[k.strip()] = x
    return out


class MetricsCache:
    """Incremental parse of training logs: every verl `step:N - k:v - ...` line plus the
    proposer reward prints. Re-reads only bytes appended since the previous call."""

    CHUNK = 8 << 20

    def __init__(self):
        self.files: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def get(self, path: Path) -> dict:
        try:
            st = path.stat()
        except OSError:
            return {"rows": [], "rewards": [], "keys": []}
        key = str(path)
        with self.lock:
            e = self.files.get(key)
            if e is None or st.st_size < e["offset"] or st.st_ino != e["ino"]:
                e = {"offset": 0, "ino": st.st_ino, "rest": "", "steps": {}, "rewards": [], "keys": set()}
                self.files[key] = e
            if st.st_size > e["offset"]:
                with path.open("rb") as f:
                    f.seek(e["offset"])
                    while True:
                        data = f.read(self.CHUNK)
                        if not data:
                            break
                        e["offset"] += len(data)
                        text = e["rest"] + data.decode("utf-8", "replace")
                        cut = max(text.rfind("\n"), text.rfind("\r"))
                        e["rest"] = text[cut + 1:] if cut >= 0 else text
                        self._consume(e, text[:cut + 1] if cut >= 0 else "")
            rows = [dict(step=s, **v) for s, v in sorted(e["steps"].items())]
            return {"rows": rows, "rewards": list(e["rewards"]), "keys": sorted(e["keys"])}

    @staticmethod
    def _consume(e: dict, text: str) -> None:
        for line in re.split(r"[\r\n]+", text):
            if "step:" in line:
                m = STEP_LINE.search(line)
                if m:
                    items = parse_step_items(m.group(2))
                    if items:
                        e["steps"].setdefault(int(m.group(1)), {}).update(items)
                        e["keys"].update(items)
                    continue
            m = FINAL_REWARD.search(line)
            if m:
                x = _num(m.group(1))
                if x is not None:
                    e["rewards"].append(x)
                continue
            m = ARM_REWARD.search(line)
            if m:
                x = _num(m.group(3))
                if x is not None:
                    e["rewards"].append(x)


def reward_log_rows(path: Path) -> List[dict]:
    """P2 arms: one row per proposer-reward call (work3/ncp_reward.py)."""
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        hist = r.get("k_hist") or {}
        n_valid = sum(hist.values()) or None
        row = {"call": r.get("call"), "mean_difficulty": r.get("mean_difficulty"),
               "mean_final": r.get("mean_final"), "valid_share": (r.get("valid") or 0) / (r.get("batch") or 1),
               "mean_w_mixed": r.get("mean_w_mixed"), "eta_mixed": r.get("eta_mixed"),
               "seconds": r.get("seconds"),
               "completion_tokens": (r.get("calls") or {}).get("completion_tokens")}
        if n_valid:
            for k, c in hist.items():
                row[f"k={k} share"] = c / n_valid
        rows.append(row)
    return rows


class History:
    """In-memory ring buffers for GPU and retriever samples (nothing is written to disk)."""

    def __init__(self, board: "Board", interval: float = 10.0, maxlen: int = 2160, gpu_fn=None):
        import collections

        self.board, self.interval = board, interval
        self.gpu = collections.deque(maxlen=maxlen)
        self.retriever: Dict[str, "collections.deque"] = collections.defaultdict(
            lambda: collections.deque(maxlen=maxlen))
        self.gpu_fn = gpu_fn or gpus
        self.reward = collections.deque(maxlen=maxlen)       # proposer-reward SGLang /metrics
        port = os.environ.get("BOARD_REWARD_PORT") or os.environ.get("REWARD_PORT") or "8101"

        def reward_fn():
            from work3.speed_probe import reward_metrics
            return reward_metrics(int(port))
        self.reward_fn = reward_fn if gpu_fn is None else None   # tests inject gpu_fn and skip HTTP
        self.lock = threading.Lock()
        self._last_ret: Dict[str, float] = {}

    def sample_once(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        g = self.gpu_fn()
        rm = self.reward_fn() if self.reward_fn else None
        with self.lock:
            if g:
                self.gpu.append({"t": now, "gpus": [{"util": x["util"], "power": x["power"],
                                                     "mem": x["mem_used"]} for x in g]})
            if rm:
                self.reward.append({"t": now, **rm})
            for run_id, _, run_dir, _, _ in self.board.specs:
                s = load_json(run_dir / "logs" / "retriever_stats.json")
                if s and s.get("updated") and s["updated"] != self._last_ret.get(run_id):
                    self._last_ret[run_id] = s["updated"]
                    self.retriever[run_id].append({"t": s["updated"], "p50": s.get("p50_ms"),
                                                   "p95": s.get("p95_ms"), "rps": s.get("requests_per_second"),
                                                   "in_flight": s.get("in_flight")})

    def start(self) -> None:
        def loop():
            while True:
                try:
                    self.sample_once()
                except Exception:  # noqa: BLE001 - sampling must never kill the board
                    pass
                time.sleep(self.interval)
        threading.Thread(target=loop, daemon=True, name="board-history").start()

    def snapshot(self) -> dict:
        with self.lock:
            return {"gpu": list(self.gpu), "retriever": {k: list(v) for k, v in self.retriever.items()},
                    "reward": list(self.reward), "interval": self.interval}


def scan_errors(text: str) -> dict:
    out = {k: len(p.findall(text)) for k, p in ERROR_PATTERNS.items()}
    names: Dict[str, int] = {}
    for n in ERROR_PATTERNS["undefined_function"].findall(text):
        names[n] = names.get(n, 0) + 1
    out["undefined_function_names"] = names
    lines = [ln for ln in clean_lines(text)
             if re.search(r"Error|FAILED|Traceback|out of memory", ln) and "undefined function" not in ln]
    out["last_lines"] = [ln[:300] for ln in lines[-3:]]
    return out


# ---------------------------------------------------------------- markdown (minimal, escaped)
def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<span title="\2" class="lnk">\1</span>', s)
    return s


def md_to_html(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r"(#{1,4})\s+(.*)", line)
        if m:
            lvl = min(len(m.group(1)) + 1, 5)
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if re.match(r"\s*([-*]|\d+\.)\s+", line):
            items = []
            while i < len(lines) and re.match(r"\s*([-*]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"\s*([-*]|\d+\.)\s+", "", lines[i], count=1))
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                        and not re.match(r"\s*([-*]|\d+\.)\s+", lines[i]):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"(#{1,4}\s|\s*\||\s*([-*]|\d+\.)\s)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


def research(root: Path = ROOT) -> dict:
    plan = (root / "research/EXPERIMENT_PLAN.md")
    log = (root / "research/DECISION_LOG.md")
    plan_md = plan.read_text(encoding="utf-8") if plan.exists() else ""
    log_md = log.read_text(encoding="utf-8") if log.exists() else ""
    entries = []
    for block in re.split(r"(?m)^## ", log_md)[1:]:
        title, _, body = block.partition("\n")
        entries.append({"title": title.strip(), "html": md_to_html(body)})
    return {"plan_html": md_to_html(plan_md), "state": load_json(root / "research/STATE.json") or {},
            "decisions": list(reversed(entries))}


# ---------------------------------------------------------------- runs
_EVAL_CACHE: Dict[str, tuple] = {}
_EVAL_LOCK = threading.Lock()


def eval_average(eval_dir: Path) -> Optional[dict]:
    """Seven-dataset EM of a finished evaluation dump (cached by mtime)."""
    files = sorted(eval_dir.glob("*.jsonl"))
    if len(files) != 1:
        return None
    key, mtime = str(files[0]), files[0].stat().st_mtime
    with _EVAL_LOCK:
        hit = _EVAL_CACHE.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    from work3 import p0_gate

    try:
        s = p0_gate.summarize_dump(files[0])
        val = {"average": s["average"], "em": s["em"], "problems": s["problems"]}
    except Exception as e:  # noqa: BLE001 - partially written dump etc.
        val = {"average": None, "error": repr(e)[:200]}
    with _EVAL_LOCK:
        _EVAL_CACHE[key] = (mtime, val)
    return val


STAGE_TEXT = {
    "prepare_train": "准备数据：生成出题种子、测试集和验证集。",
    "prepare_test": "准备数据：生成出题种子、测试集和验证集。",
    "prepare_val": "准备数据：生成出题种子、测试集和验证集。",
    "challenger": "出题者训练：出题者从维基百科文档出发，边搜索边出多跳题；每道题让解题者答 5 次，"
                  "按答对次数给出题奖励，再用强化学习（HRPO）更新出题者。共 50 步，每步 256 道题。",
    "gen_data": "生成题池：用刚训练好的出题者为解题者生成约 3.4 万道训练题（每个种子采样 5 次，取格式最好的一道）。",
    "pool_check": "题池检查：确认题池里存的是解题者提示（防止 pandas 版本问题写坏数据）。",
    "solver": "解题者训练：解题者在新题池上用 GRPO 训练 50 步，每题 5 次多轮“思考→搜索→作答”。",
    "convert": "转换检查点：把解题者的 FSDP 检查点转成 HuggingFace 格式，供下一轮出题者打分和评测使用。",
    "eval": "评测：在 7 个 QA 数据集（共 51,713 题）上用贪心解码测 EM，是 P0 门槛的依据。",
    "pool": "P1 诊断：抽题池中的题，每题解答 5 次，每次再做 2 次只打乱文档顺序的重放，测量成败中“运气”的占比。",
    "parity": "一致性检查：核对 P1 拼出的提示与上游 verl 逐字一致（只用 CPU）。",
}


def stage_type(name: str) -> str:
    if name.startswith("prepare_"):
        return name
    return re.sub(r"^iter\d_|\d$", "", name)


LANE_SEGMENTS = [("出题者训练", ("challenger",)), ("生成题池", ("gen_data", "pool_check")),
                 ("解题者训练", ("solver", "convert")), ("评测", ("eval",))]


def _segment(label: str, members: List[dict], extra: Optional[dict] = None) -> dict:
    states = [m["state"] for m in members]
    if not members:
        state = "pending"
    elif "failed" in states:
        state = "failed"
    elif "stalled" in states:
        state = "stalled"
    elif "running" in states or ("done" in states and "pending" in states):
        state = "running"
    elif all(s == "done" for s in states):
        state = "done"
    else:
        state = "pending"
    est = sum(m["est_seconds"] for m in members)
    frac = (sum(m["est_seconds"] * m["fraction"] for m in members) / est) if est else 0.0
    seg = {"label": label, "state": state, "est_seconds": est, "fraction": frac,
           "seconds": sum(m["seconds"] or 0 for m in members) or None,
           "stages": [m["name"] for m in members]}
    seg.update(extra or {})
    return seg


def build_lanes(rows: List[dict], kind: str, evals: dict, has_summary: bool) -> List[dict]:
    """Group raw stages into a few readable lanes: 准备 / 第 t 轮 (4 segments) / 收尾."""
    by = {r["name"]: r for r in rows}
    lanes = []
    if kind == "p0":
        prep = [by[n] for n in ("prepare_train", "prepare_test", "prepare_val") if n in by]
        if prep:
            lanes.append({"label": "准备", "segments": [_segment("准备数据", prep)]})
        for t in (1, 2, 3):
            segs = []
            for label, types in LANE_SEGMENTS:
                names = [f"eval{t}"] if types == ("eval",) else [f"iter{t}_{x}" for x in types]
                members = [by[n] for n in names if n in by]
                if members:
                    extra = {"em": (evals.get(str(t)) or {}).get("average")} if types == ("eval",) else None
                    segs.append(_segment(label, members, extra))
            if segs:
                lanes.append({"label": f"第 {t} 轮", "segments": segs})
        tail = []
        if "eval0" in by:
            tail.append(_segment("基座模型评测（参考）", [by["eval0"]], {"em": (evals.get("0") or {}).get("average")}))
        tail.append({"label": "门槛判定", "state": "done" if has_summary else "pending", "est_seconds": 60,
                     "fraction": 1.0 if has_summary else 0.0, "seconds": None, "stages": []})
        lanes.append({"label": "收尾", "segments": tail})
    else:
        segs = [_segment(r["label"], [r]) for r in rows]
        segs.append({"label": "汇总判定", "state": "done" if has_summary else "pending", "est_seconds": 60,
                     "fraction": 1.0 if has_summary else 0.0, "seconds": None, "stages": []})
        lanes.append({"label": "诊断", "segments": segs})
    return lanes


def training_fraction(stage: str, bar: Optional[dict]) -> Optional[float]:
    """Share of this stage's 50 steps done. Proposer iterations resume (total 50/100/150)."""
    if not bar or not bar.get("total"):
        return None
    m = re.match(r"iter(\d)_challenger$", stage)
    base = 50 * (int(m.group(1)) - 1) if m else 0
    span = bar["total"] - base
    if span <= 0:
        return None
    return max(0.0, min(1.0, (bar["n"] - base) / span))


def stage_rows(run_dir: Path, plan, now: float) -> List[dict]:
    tpath = run_dir / "timings.txt"
    timings = parse_timings(tpath)
    tmtime = tpath.stat().st_mtime if tpath.exists() else 0.0
    logs = run_dir / "logs"
    rows = []
    for name, label, hours in plan:
        log = logs / f"{name}.log"
        mtime = log.stat().st_mtime if log.exists() else None
        t = timings.get(name)
        end = t["end"] if t and t["end"] is not None else tmtime   # old two-field timings lines
        if t and (mtime is None or mtime <= end + 5):
            if t["rc"] is None:   # p0-r1 lines carry no exit code: infer from the log tail
                failed = bool(re.search(r"Traceback \(most recent call last\)|Error executing job",
                                        tail_text(log, 20_000)))
            else:
                failed = t["rc"] != 0
            state = "failed" if failed else "done"
        elif mtime is not None:
            state = "running" if now - mtime < STALL_SECONDS else "stalled"
        else:
            state = "pending"
        rows.append({"name": name, "label": label, "state": state, "default_hours": hours,
                     "seconds": t["seconds"] if t else None, "log_mtime": mtime})
    return rows


def run_status(run_id: str, title: str, run_dir: Path, plan, kind: str, now: float) -> dict:
    rows = stage_rows(run_dir, plan, now)
    active = next((r for r in rows if r["state"] in ("running", "stalled", "failed")), None)
    current = None
    if active:
        text = tail_text(run_dir / "logs" / f"{active['name']}.log")
        bars = tqdm_progress(text)
        steps = step_metrics(text)
        current = {"stage": active["name"], "label": active["label"], "state": active["state"],
                   "started": None, "bars": bars, "last_step": steps[-1] if steps else None,
                   "updated": active["log_mtime"], "_steps": steps}
        if active["state"] in ("running", "stalled") and active["log_mtime"]:
            current["idle_seconds"] = round(now - active["log_mtime"])
    # Duration estimate per stage: observed mean of the same stage type, else the guide's default.
    observed: Dict[str, List[int]] = {}
    for r in rows:
        if r["state"] == "done" and r["seconds"]:
            observed.setdefault(stage_type(r["name"]), []).append(r["seconds"])
    last_end = max((t["end"] for t in parse_timings(run_dir / "timings.txt").values() if t["end"]), default=None)
    for r in rows:
        typ = stage_type(r["name"])
        r["est_seconds"] = (sum(observed[typ]) / len(observed[typ])) if typ in observed else r["default_hours"] * 3600
        r["fraction"] = 1.0 if r["state"] == "done" else 0.0
        if current and r["name"] == current["stage"] and r["state"] in ("running", "stalled"):
            bar = current["bars"].get("Training Progress")
            frac = training_fraction(r["name"], bar)
            if bar:
                mm = re.match(r"iter(\d)_challenger$", r["name"])
                base = 50 * (int(mm.group(1)) - 1) if mm else 0
                current["steps_done"], current["steps_total"] = max(0, bar["n"] - base), max(0, bar["total"] - base)
            if frac is None and last_end:                      # no step bar: elapsed since previous stage ended
                frac = min(0.95, max(0.0, (now - last_end) / r["est_seconds"]))
            # Steps get slower as the proposer improves (more valid questions to score), so the
            # remaining time of a training stage uses the mean of its latest five step times.
            recent = [s["timing_s/step"] for s in current.pop("_steps", []) if "timing_s/step" in s][-5:]
            if recent and current.get("steps_total") and frac is not None and frac < 1:
                left = (current["steps_total"] - current["steps_done"]) * sum(recent) / len(recent)
                r["est_seconds"] = left / (1 - frac)
                current["eta_basis"] = f"按最近 {len(recent)} 步平均 {round(sum(recent) / len(recent))} 秒/步"
            current["fraction"] = frac
            current["elapsed_seconds"] = round(now - last_end) if last_end else None
            r["fraction"] = frac or 0.0
    total = sum(r["est_seconds"] for r in rows) or 1.0
    done_part = sum(r["est_seconds"] * r["fraction"] for r in rows)
    remaining = sum(r["est_seconds"] * (1 - r["fraction"]) for r in rows if r["state"] != "done")
    if current:
        current.pop("_steps", None)
        current["explain"] = STAGE_TEXT.get(stage_type(current["stage"]), "")
    # errors over all logs (tails)
    errors = {"traceback": 0, "failed": 0, "oom": 0, "undefined_function": 0,
              "undefined_function_names": {}, "last_lines": []}
    for log in sorted((run_dir / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime):
        e = scan_errors(tail_text(log))
        for k in ("traceback", "failed", "oom", "undefined_function"):
            errors[k] += e[k]
        for n, c in e["undefined_function_names"].items():
            errors["undefined_function_names"][n] = errors["undefined_function_names"].get(n, 0) + c
        errors["last_lines"] += [f"{log.name}: {ln}" for ln in e["last_lines"]]
    errors["last_lines"] = errors["last_lines"][-5:]
    evals = {}
    if kind == "p0":
        for t in (0, 1, 2, 3):
            d = run_dir / "eval" / f"iter{t}"
            if d.is_dir() and any(r["name"] == f"eval{t}" and r["state"] == "done" for r in rows):
                evals[str(t)] = eval_average(d)
    summary_name = "p0_summary.json" if kind == "p0" else "p1_summary.json"
    summary = load_json(run_dir / summary_name)
    started = any(r["state"] != "pending" for r in rows)
    return {"id": run_id, "title": title, "dir": str(run_dir), "kind": kind, "exists": run_dir.is_dir(),
            "started": started, "stages": rows, "current": current,
            "eta_seconds": round(remaining) if run_dir.is_dir() else None,
            "overall_fraction": done_part / total, "finish_at": (now + remaining) if started else None,
            "lanes": build_lanes(rows, kind, evals, bool(summary)),
            "errors": errors, "evals": evals, "threshold": P0_THRESHOLD,
            "summary": ({k: summary.get(k) for k in ("verdict", "problems", "pooled", "primary",
                                                     "threshold_iter3_average")} if summary else None),
            "retriever": load_json(run_dir / "logs" / "retriever_stats.json"),
            "pool_checks": {p.stem: (load_json(p) or {}).get("ok") for p in sorted(run_dir.glob("pool_check_iter*.json"))}}


class Board:
    def __init__(self, drzero_dir: Optional[str], arm_root: Optional[str], root: Path = ROOT):
        self.root = root
        dz = Path(drzero_dir) if drzero_dir else None
        p0 = os.environ.get("BOARD_P0_OUT") or (str(dz / "work3_p0") if dz else None)
        p1 = os.environ.get("BOARD_P1_OUT") or (str(dz / "work3_p1") if dz else None)
        ar = Path(arm_root) if arm_root else (dz.parent / "drzero_p2" if dz else None)
        self.specs = []
        if p0:
            self.specs.append(("p0", "P0 复现 Dr. Zero", Path(p0), P0_STAGES, "p0"))
        if p1:
            self.specs.append(("p1", "P1 噪声诊断", Path(p1), P1_STAGES, "p1"))
        if ar:
            for arm in ("beta", "ncp"):
                self.specs.append((f"p2-{arm}", f"P2 {arm} 臂", ar / arm / "work3_p0", ARM_STAGES, "p0"))
                self.specs.append((f"p2-{arm}-p1", f"P2 {arm} 臂 · 题池 3 诊断", ar / arm / "work3_p1",
                                   P1_ARM_STAGES, "p1"))
        self.p2_summary = ar / "p2_summary.json" if ar else None
        self.metrics_cache = MetricsCache()
        self.history = History(self)

    def _spec(self, run_id: str):
        return next((s for s in self.specs if s[0] == run_id), None)

    def metric_stages(self, run_id: str) -> Optional[dict]:
        """Training / evaluation logs of a run that exist and carry metrics."""
        spec = self._spec(run_id)
        if not spec:
            return None
        names = [n for n, _, _ in spec[3] if ("challenger" in n or "solver" in n or n.startswith("eval"))
                 and (spec[2] / "logs" / f"{n}.log").exists()]
        labels = {n: lbl for n, lbl, _ in spec[3]}
        return {"run": run_id, "stages": [{"name": n, "label": labels[n]} for n in names],
                "reward_log": (spec[2] / "reward_log.jsonl").exists()}

    def metrics(self, run_id: str, stage: str, compare: bool = False) -> Optional[dict]:
        spec = self._spec(run_id)
        if not spec or stage not in {n for n, _, _ in spec[3]}:      # only known stage names
            return None
        m = re.match(r"iter(\d)_(\w+)$", stage)
        targets = [stage]
        if compare and m:
            targets = [f"iter{t}_{m.group(2)}" for t in (1, 2, 3)]
        series = {}
        for name in targets:
            log = spec[2] / "logs" / f"{name}.log"
            if not log.exists():
                continue
            data = self.metrics_cache.get(log)
            mm = re.match(r"iter(\d)_challenger$", name)
            base = 50 * (int(mm.group(1)) - 1) if mm else 0     # proposer steps continue across iterations
            data["step_offset"] = base
            series[name] = data
        return {"run": run_id, "stage": stage, "compare": compare and bool(m), "series": series}

    def reward_log(self, run_id: str) -> Optional[dict]:
        spec = self._spec(run_id)
        if not spec:
            return None
        return {"run": run_id, "rows": reward_log_rows(spec[2] / "reward_log.jsonl")}

    def state(self) -> dict:
        now = time.time()
        runs = [run_status(i, t, d, plan, k, now) for i, t, d, plan, k in self.specs]
        p2 = load_json(self.p2_summary) if self.p2_summary else None
        return {"now": now, "runs": runs, "gpus": gpus(), "p2": p2,
                "roadmap": roadmap(runs, p2, load_json(self.root / "research/STATE.json") or {})}

    def log_tail(self, run_id: str, lines: int = 80) -> Optional[dict]:
        spec = next((s for s in self.specs if s[0] == run_id), None)
        if not spec:
            return None
        logs = sorted((spec[2] / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            return {"file": None, "lines": []}
        return {"file": logs[0].name, "lines": [ln[:400] for ln in clean_lines(tail_text(logs[0], 200_000))[-lines:]]}


ROADMAP_GATES = {
    "P0": "第 3 轮 7 数据集平均 EM ≥ 0.306（论文 0.326）",
    "P1": "k=1 题的噪声占比 η ≥ 0.35，且比 k=3–4 的题高 ≥ 0.10",
    "P2": "NCP 比 Dr. Zero 高 ≥ 2 pp、比 Beta 高 ≥ 1 pp，且第 3 轮不下降",
    "M": "P2 通过后另行登记：多种子、7B、消融、推广",
}
VERDICT_TEXT = {"PASS": ("通过", "done"), "FAIL": ("未通过", "failed"), "INVALID": ("无效，需重跑", "stalled"),
                "GO": ("通过（GO）", "done"), "NO-GO": ("未通过（NO-GO）", "failed")}


def roadmap(runs: List[dict], p2: Optional[dict], state: dict) -> List[dict]:
    by = {r["id"]: r for r in runs}

    def phase(key, title, what, run_ids, verdict=None, result=None):
        rs = [by[i] for i in run_ids if i in by]
        if verdict in VERDICT_TEXT:
            status, cls = VERDICT_TEXT[verdict]
        elif any(r["started"] for r in rs):
            done = [r["overall_fraction"] for r in rs if r["started"]]
            status, cls = f"进行中 {round(100 * sum(done) / max(1, len(rs)))}%", "running"
        else:
            status, cls = "未开始", "pending"
        return {"key": key, "title": title, "what": what, "gate": ROADMAP_GATES.get(key, ""),
                "status": status, "state": cls, "result": result}

    p0 = by.get("p0") or {}
    p0_em = [(t, e.get("average")) for t, e in sorted((p0.get("evals") or {}).items())
             if t != "0" and e and e.get("average") is not None]
    p0_res = f"第 {p0_em[-1][0]} 轮 EM {p0_em[-1][1]:.3f}" if p0_em else None
    p1 = by.get("p1") or {}
    pooled = ((p1.get("summary") or {}).get("pooled") or {})
    p1_res = (f"η(k=1) {pooled['eta_k1']:.2f}，差值 {pooled['diff']:.2f}"
              if pooled.get("eta_k1") is not None and pooled.get("diff") is not None else None)
    prim = (p2 or {}).get("primary") or {}
    p2_res = (f"NCP−Dr.Zero {prim['ncp_minus_drzero'] * 100:+.1f} pp"
              if prim.get("ncp_minus_drzero") is not None else None)
    venue = (state.get("target_venue") or {})
    return [
        phase("P0", "P0 复现基线", "复现 Dr. Zero 3B 三轮自我进化", ["p0"],
              (p0.get("summary") or {}).get("verdict"), p0_res),
        phase("P1", "P1 噪声诊断", "测题池里“运气”的占比，证伪核心假设", ["p1"],
              (p1.get("summary") or {}).get("verdict"), p1_res),
        phase("P2", "P2 方法对比", "Dr. Zero / Beta 平滑 / NCP 三臂对比", ["p2-beta", "p2-ncp"],
              (p2 or {}).get("verdict"), p2_res),
        phase("M", "主实验", "多种子、7B、消融、推广到其他自博弈", []),
        {"key": "S", "title": "投稿", "what": f"{venue.get('primary', 'ICML 2027')}（备选 {venue.get('alternative', 'SIGIR 2027')}）",
         "gate": f"{venue.get('decide_by', '2026-12-15')} 前确定投哪个会", "status": "预计 2027-01 下旬",
         "state": "pending", "result": None},
    ]


def gpus() -> Optional[List[dict]]:
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                              "power.draw,power.limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    out = []
    for line in res.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) != 7:
            continue
        num = lambda x: float(x) if re.fullmatch(r"[\d.]+", x) else None
        out.append({"index": p[0], "name": p[1], "util": num(p[2]), "mem_used": num(p[3]),
                    "mem_total": num(p[4]), "power": num(p[5]), "power_limit": num(p[6])})
    return out


def make_handler(board: Board):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body, kind: str) -> None:
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def json(self, value) -> None:
            self.send(200, json.dumps(value, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/research":
                    self.json(research(board.root))
                elif url.path == "/api/briefing":
                    path = board.root / "research" / "briefing.html"
                    body = path.read_bytes() if path.exists() else "<p>research/briefing.html 不存在</p>".encode()
                    self.send(200, body, "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self.json(board.state())
                elif url.path == "/api/log":
                    t = board.log_tail(q.get("run", [""])[0])
                    self.json(t) if t is not None else self.send(404, "unknown run", "text/plain; charset=utf-8")
                elif url.path == "/api/stages":
                    t = board.metric_stages(q.get("run", [""])[0])
                    self.json(t) if t is not None else self.send(404, "unknown run", "text/plain; charset=utf-8")
                elif url.path == "/api/metrics":
                    t = board.metrics(q.get("run", [""])[0], q.get("stage", [""])[0],
                                      q.get("compare", ["0"])[0] == "1")
                    self.json(t) if t is not None else self.send(404, "unknown run or stage",
                                                                 "text/plain; charset=utf-8")
                elif url.path == "/api/rewardlog":
                    t = board.reward_log(q.get("run", [""])[0])
                    self.json(t) if t is not None else self.send(404, "unknown run", "text/plain; charset=utf-8")
                elif url.path == "/api/history":
                    self.json(board.history.snapshot())
                else:
                    self.send(404, "not found", "text/plain; charset=utf-8")
            except Exception as e:  # noqa: BLE001 - keep the board alive
                self.send(500, f"board error: {e!r}", "text/plain; charset=utf-8")

        def log_message(self, *args) -> None:
            pass
    return Handler


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 only on a trusted internal network")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--drzero-dir", default=os.environ.get("DRZERO_DIR"))
    p.add_argument("--arm-root", default=os.environ.get("ARM_ROOT"))
    a = p.parse_args(argv)
    board = Board(a.drzero_dir, a.arm_root)
    board.history.start()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(board))
    print(f"Work 3 board on http://{a.host}:{a.port}/  (runs: "
          + ", ".join(str(s[2]) for s in board.specs) + "); Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
