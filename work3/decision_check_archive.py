"""Close an interrupted B0 from its saved outputs without executing any games."""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from work3.decision_check_preflight import MIN_ELIGIBLE, MIN_PER_TYPE, MIN_TYPES, PER_TASK, RUN_ID, TASK_TYPES
from work3.wiki_rsi_progress import atomic_json, progress_root

MODULE = b"work3.decision_check_preflight"


def live_runners(run_id, proc_root=Path("/proc")):
    target = run_id.encode("ascii")
    runners = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not any(a == b"-m" and b == MODULE for a, b in zip(argv, argv[1:])):
            continue
        if b"--id=" + target in argv or any(a == b"--id" and b == target for a, b in zip(argv, argv[1:])):
            runners.append(int(entry.name))
    return sorted(runners)


def partial_bounds(raw, done, total):
    records = [json.loads(line) for line in raw.splitlines()]
    if not 0 <= done < total <= len(TASK_TYPES) * PER_TASK or len(records) != done:
        raise ValueError("B0 progress and saved record count disagree")
    checked = Counter(row["type"] for row in records)
    if set(checked) - TASK_TYPES or max(checked.values(), default=0) > PER_TASK:
        raise ValueError("saved task counts are inconsistent with the frozen cap")
    if len({row["sha256"] for row in records}) != done or len({row["file"] for row in records}) != done:
        raise ValueError("duplicate saved B0 game records")
    allowed = {"qualified", "no_grounded_choice", "no_replayable_outcome_gap", "planner_did_not_finish"}
    reasons = Counter(row["reason"] for row in records)
    if set(reasons) - allowed:
        raise ValueError("unknown saved B0 outcome")
    eligible = Counter(row["type"] for row in records if row["reason"] == "qualified")
    remaining = total - done
    upper = sum(eligible.values()) + remaining
    type_upper = sum(eligible[task] + min(remaining, PER_TASK - checked[task]) >= MIN_PER_TYPE for task in TASK_TYPES)
    return {"tested_games": done, "planned_games": total, "eligible_games": sum(eligible.values()),
            "reason_counts": dict(reasons), "checked_by_type": dict(checked), "eligible_by_type": dict(eligible),
            "eligible_upper_bound": upper, "eligible_type_upper_bound": type_upper,
            "research_gate": "UNREACHABLE" if upper < MIN_ELIGIBLE or type_upper < MIN_TYPES else "UNDETERMINED"}


def archive(directory, now, proc_root=Path("/proc")):
    progress_path = directory / "progress.json"
    incomplete = directory / "incomplete.json"
    done_file = directory / "wiki_rsi_decision_check_summary.json"
    if done_file.exists():
        raise ValueError("B0 already wrote a complete summary")
    saved = json.loads(progress_path.read_text(encoding="utf-8"))
    phase = saved["phases"]["decision_check"]
    if incomplete.exists():
        existing = json.loads(incomplete.read_text(encoding="utf-8"))
        if existing.get("closure") != "stale_run_archive":
            raise ValueError("B0 already wrote an INCOMPLETE record")
        if phase["state"] == "failed":
            return existing
    if saved["schema_version"] != 1 or phase["state"] not in {"running", "stalled"}:
        raise ValueError("archive requires an interrupted old running or stalled B0")
    if now - phase.get("heartbeat_at", phase["updated_at"]) <= 60:
        raise ValueError("B0 heartbeat is still fresh")
    pids = live_runners(directory.name, proc_root)
    if pids:
        raise ValueError(f"B0 processes still present for {directory.name}: {pids}; stop and recheck this run first")
    raw = (directory / "decision_checks.jsonl").read_bytes()
    counts = partial_bounds(raw, phase["done"], phase["total"])
    if counts["eligible_games"] != phase["metrics"]["eligible_games"]:
        raise ValueError("B0 progress and saved qualified count disagree")
    result = {"verdict": "INCOMPLETE", "closure": "stale_run_archive", "run_id": directory.name,
              "repo_revision": phase["commit"], "archived_at": now,
              "rollouts_sha256": hashlib.sha256(raw).hexdigest(), "original_progress": phase.copy(), **counts}
    atomic_json(incomplete, result)
    phase.update(state="failed", detail="decision_check_archived", updated_at=now, heartbeat_at=now)
    saved["updated_at"] = now
    atomic_json(progress_path, saved)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(RUN_ID, args.id) or os.name != "posix" or not Path("/proc/self").exists():
        parser.error("archive requires a B0 run id and a Linux process table")
    directory = progress_root() / args.id
    if not directory.is_dir() or directory.is_symlink():
        parser.error("run directory must already exist")
    try:
        print(json.dumps(archive(directory, int(time.time())), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
