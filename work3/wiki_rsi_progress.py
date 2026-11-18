"""Atomic status snapshots shared by the Wikipedia experiment and read-only board."""

import json
import os
import time
from pathlib import Path

PHASES = ("audit", "prepare", "preflight", "smoke", "replay", "w2a_prepare", "w2a")
STATES = ("running", "done", "failed", "blocked", "stalled")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "wiki_rsi"


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
        self.data = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists()
                     else {"schema_version": 1, "phases": {}})

    def update(self, state, done=0, total=0, *, detail="", metrics=None):
        now = int(time.time())
        phases = self.data["phases"]
        start = phases.get(self.phase, {}).get("started_at", now)
        if state == "running" and phases.get(self.phase, {}).get("state") != "running":
            start = now
        phases[self.phase] = {
            "state": state, "done": done, "total": total, "detail": detail,
            "updated_at": now, "started_at": start, "metrics": metrics or {},
        }
        self.data["updated_at"] = now
        atomic_json(self.path, self.data)


def board_runs(root, now, limit=5):
    runs = []
    if not root.is_dir():
        return runs
    for directory in root.iterdir():
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
                if state == "running" and now - row["updated_at"] > 1200:
                    state = "stalled"
                phases[phase] = {key: row[key] for key in
                                 ("done", "total", "started_at", "updated_at")}
                phases[phase].update(
                    state=state, detail=str(row["detail"])[:80],
                    metrics={key: value for key, value in row["metrics"].items()
                             if key in {"hotpotqa", "2wikimultihopqa", "wall_seconds",
                                        "hotpotqa_added", "2wikimultihopqa_added",
                                        "queries",
                                        "generated_tokens", "prompt_tokens", "valid_triplets", "completion_jobs"}
                             and isinstance(value, (int, float))})
            runs.append({"id": directory.name, "updated_at": data["updated_at"], "phases": phases})
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            continue
    return sorted(runs, key=lambda run: (-run["updated_at"], run["id"]))[:limit]
