"""Atomic status snapshots shared by the Wikipedia experiment and read-only board."""

import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PHASES = ("audit", "prepare", "preflight", "smoke", "replay", "w2a_prepare", "w2a",
          "w2_contract_prepare", "w2_contract", "w2_budget_prepare", "w2_budget",
          "w2_answer_prepare", "w2_answer", "w3_candidates_prepare", "w3_candidates",
          "w3_candidate_audit_prepare", "w3_candidate_audit",
          "w3_coverage_audit_prepare", "w3_coverage_audit",
          "w4_repair_pool_prepare", "w4_repair_pool", "w4_repair_audit",
          "w4_repair_answer_prepare", "w4_repair_answer", "w5_natural_prepare", "w5_natural",
          "w5_natural_query_prepare", "w5_natural_query",
          "w6_action_pilot_prepare", "w6_action_pilot",
          "w6_action_train_prepare", "w6_action_train", "w6_overnight_prepare", "w6_overnight",
          "w6_query_diagnosis", "w7_two_query", "w7_live_prepare", "w7_live", "switch_prepare", "switch")
STATES = ("running", "done", "failed", "blocked", "stalled")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs" / "wiki_rsi"
SUMMARY_PATHS = {"audit": "w0/wiki_rsi_input_summary.json", **{
    phase: f"wiki_rsi_{phase}_summary.json" for phase in PHASES if phase != "audit"
}}


def progress_root():
    return Path(os.environ.get("WIKI_RSI_ROOT") or DEFAULT_ROOT).resolve()


def atomic_json(path, data, mode=0o600):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode),
                   "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


class Progress:
    def __init__(self, run_dir, phase):
        self.path = run_dir / "progress.json"
        self.phase = phase
        self.lock = threading.Lock()
        try:
            self.commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL,
                text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            self.commit = None
        self.data = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists()
                     else {"schema_version": 1, "phases": {}})

    def update(self, state, done=0, total=0, *, detail="", metrics=None, dataset=None):
        now = int(time.time())
        with self.lock:
            phases = self.data["phases"]
            previous = phases.get(self.phase, {})
            start = previous.get("started_at", now)
            if state == "running" and previous.get("state") != "running":
                start = now
            phases[self.phase] = {
                "state": state, "done": done, "total": total, "detail": detail, "dataset": dataset,
                "updated_at": now, "started_at": start, "heartbeat_at": now,
                "commit": self.commit, "metrics": metrics or {},
                "action_started_at": previous.get("action_started_at", now)
                    if previous.get("state") == state and previous.get("detail") == detail else now,
            }
            self.data["updated_at"] = now
            atomic_json(self.path, self.data)

    @contextmanager
    def heartbeat(self, interval=3):
        event = threading.Event()

        def loop():
            while not event.wait(interval):
                with self.lock:
                    row = self.data["phases"].get(self.phase, {})
                    if row.get("state") == "running":
                        row["heartbeat_at"] = int(time.time())
                        atomic_json(self.path, self.data)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        try:
            yield
        finally:
            event.set()
            thread.join()


def commit_from_summary(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("repo_revision") or data.get("provenance", {}).get("repo_revision")
    except (OSError, ValueError, AttributeError):
        return None


def public_commit(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{7,40}", value) else None


def run_view(run_id, updated_at, phases):
    start = [row["started_at"] for row in phases.values() if row["started_at"] is not None]
    newest = sorted(phases.values(), key=lambda row: row["updated_at"], reverse=True)
    active = any(row["state"] in {"running", "stalled"} for row in newest)
    return {"id": run_id, "updated_at": updated_at, "phases": phases,
            "started_at": min(start) if start else None, "finished_at": None if active else updated_at,
            "commit": next((row["commit"] for row in newest if row["commit"]), None)}


def legacy_audits(root):
    if not root or not root.is_dir():
        return []
    runs = []
    for folder in root.glob("wiki_rsi_audit-*"):
        path = folder / "wiki_rsi_input_summary.json"
        if folder.is_symlink() or path.is_symlink():
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary["protocol"] != "rsi_wikipedia_input_audit_v1":
                continue
            ready = summary["ready_for_replay_preparation"]
            stamp = int(path.stat().st_mtime)
            metrics = {ds: summary["by_dataset"][ds].get("selected_missing_with_source", 0)
                       for ds in ("hotpotqa", "2wikimultihopqa")}
            row = {"state": "done" if ready else "blocked", "detail": "ready" if ready else "insufficient_candidates",
                   "done": 1, "total": 1, "updated_at": stamp, "started_at": None,
                   "commit": public_commit(summary["provenance"].get("repo_revision")), "metrics": metrics}
            runs.append(run_view(folder.name, stamp, {"audit": row}))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return runs


def board_runs(root, now, limit=20, legacy_root=None):
    runs = []
    for directory in root.iterdir() if root.is_dir() else ():
        path = directory / "progress.json"
        if directory.is_symlink() or path.is_symlink() or not directory.is_dir():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["schema_version"] != 1 or not isinstance(data["updated_at"], int):
                continue
            phases = {}
            for phase in PHASES:
                row = data["phases"].get(phase)
                if not row or row["state"] not in STATES:
                    continue
                if not all(isinstance(row[key], int)
                           for key in ("done", "total", "started_at", "updated_at")):
                    continue
                state = row["state"]
                heartbeat = row.get("heartbeat_at")
                timed_out = now - heartbeat > 60 if isinstance(heartbeat, int) else now - row["updated_at"] > 1200
                if state == "running" and timed_out:
                    state = "stalled"
                commit = public_commit(row.get("commit"))
                if commit is None and state in {"done", "blocked"}:
                    commit = public_commit(commit_from_summary(directory / SUMMARY_PATHS[phase]))
                phases[phase] = {key: row[key] for key in
                                 ("done", "total", "started_at", "updated_at")}
                dataset = row.get("dataset")
                phases[phase].update(
                    state=state, detail=str(row["detail"])[:80],
                    commit=commit, heartbeat_at=heartbeat,
                    dataset=dataset if dataset in ("hotpotqa", "2wikimultihopqa") else None,
                    action_started_at=row.get("action_started_at", row["updated_at"]),
                    metrics={key: value for key, value in row["metrics"].items()
                             if key in {"hotpotqa", "2wikimultihopqa", "wall_seconds",
                                        "hotpotqa_added", "2wikimultihopqa_added",
                                        "queries", "inference_gpus", "legacy_proposals", "corpus_rows",
                                        "generated_tokens", "prompt_tokens", "valid_triplets", "completion_jobs"}
                             and isinstance(value, (int, float))})
            runs.append(run_view(directory.name, data["updated_at"], phases))
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            continue
    existing = {run["id"] for run in runs}
    runs.extend(run for run in legacy_audits(legacy_root) if run["id"] not in existing)
    return sorted(runs, key=lambda run: (-run["updated_at"], run["id"]))[:limit]
