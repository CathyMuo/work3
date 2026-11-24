"""Bounded, CPU-only readiness probe for decision-relevant checks in ALFWorld."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from time import monotonic

from work3.wiki_rsi_progress import Progress, atomic_json, progress_root

VERSIONS = {"alfworld": "0.4.2", "textworld": "1.6.2"}
TASK_TYPES = {
    "pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
}
PER_TASK = 40
HORIZON = 40
CHECKS_PER_GAME = 2
HELDOUT_PER_SPLIT = 100
MIN_ELIGIBLE = 100
MIN_TYPES = 3
MIN_PER_TYPE = 10
RUN_SECONDS = 7200
GAME_SECONDS = 90
RUN_ID = r"decision-check-[a-z0-9][a-z0-9-]{0,55}"
ENTITY = re.compile(r"\b[a-z][a-z_-]* \d+\b", re.I)
TAKE = re.compile(r"take (.+) from (.+)\Z")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def select_games(root):
    groups = defaultdict(list)
    for game in root.glob("*/*/game.tw-pddl"):
        relative_path = game.relative_to(root)
        relative = relative_path.as_posix()
        if "movable" in relative or "Sliced" in relative:
            continue
        task = relative_path.parts[0].partition("-")[0]
        raw = game.read_bytes()
        if task in TASK_TYPES and json.loads(raw).get("solvable") is True:
            groups[task].append({"file": relative, "type": task, "sha256": sha(raw)})
    result, used = [], set()
    for task in sorted(groups):
        ordered = sorted(groups[task], key=lambda row: sha(("work3-b0:" + row["file"]).encode()))
        for row in ordered:
            if row["sha256"] not in used:
                used.add(row["sha256"])
                result.append(row)
    return result


def heldout_seal(train, roots):
    used = {row["sha256"] for row in train}
    heldout = {}
    for label, path in roots.items():
        candidates = sorted(select_games(path), key=lambda row: sha(("work3-b0-heldout:" + row["file"]).encode()))
        chosen = [row for row in candidates if row["sha256"] not in used][:HELDOUT_PER_SPLIT]
        if len(chosen) != HELDOUT_PER_SPLIT:
            raise ValueError(f"{label}: expected {HELDOUT_PER_SPLIT} disjoint, solvable official heldout games")
        heldout[label] = chosen
        used.update(row["sha256"] for row in chosen)
    return heldout


def snapshot(state):
    return (state["feedback"], tuple(state["admissible_commands"]),
            tuple(sorted(str(fact) for fact in state["facts"])), bool(state["won"]))


def step(env, state, action):
    if action not in state["admissible_commands"]:
        raise ValueError(f"planner or replay emitted an inadmissible action: {action}")
    return env.step(action)[0]


def outcome(state, steps):
    return {"won": bool(state["won"]), "steps_after_check": steps}


def fork_candidate(before, after, plan, taken, seen):
    if not plan:
        return None
    preferred = TAKE.fullmatch(plan[0])
    if preferred is None or preferred[2] != taken[5:]:
        return None
    observed = set(name.casefold() for name in ENTITY.findall(after["feedback"])) - seen
    if preferred[1].casefold() not in observed:
        return None
    alternatives = []
    for action in after["admissible_commands"]:
        parsed = TAKE.fullmatch(action)
        if action != plan[0] and parsed and parsed[2] == preferred[2] and parsed[1].casefold() in observed:
            alternatives.append(action)
    if not alternatives:
        return None
    return {"before": snapshot(before), "after": snapshot(after), "preferred": plan[0],
            "alternative": sorted(alternatives)[0], "revealed": sorted(observed)}


def probe(env):
    state = env.reset()
    prefix, candidates = [], []
    seen = {name.casefold() for name in ENTITY.findall(state["feedback"])}
    while not state["won"] and len(prefix) < HORIZON and state["policy_commands"]:
        action = state["policy_commands"][0]
        upcoming = step(env, state, action)
        seen.update(name.casefold() for name in ENTITY.findall(action))
        if action.startswith("open ") and len(candidates) < CHECKS_PER_GAME:
            candidate = fork_candidate(state, upcoming, upcoming["policy_commands"], action, seen)
            if candidate:
                candidates.append({**candidate, "offset": len(prefix), "check": action})
        seen.update(name.casefold() for name in ENTITY.findall(upcoming["feedback"]))
        prefix.append(action)
        state = upcoming
    if not state["won"]:
        return {"reason": "planner_did_not_finish"}
    for item in candidates:
        offset = item["offset"]
        state = env.reset()
        for action in prefix[:offset]:
            state = step(env, state, action)
        if snapshot(state) != item["before"]:
            continue
        after = step(env, state, item["check"])
        if snapshot(after) != item["after"]:
            continue
        state = step(env, after, item["alternative"])
        alt_steps = 1
        cap = HORIZON - offset - 1
        while not state["won"] and alt_steps < cap and state["policy_commands"]:
            state = step(env, state, state["policy_commands"][0])
            alt_steps += 1
        best_steps = len(prefix) - offset - 1
        if state["won"] and alt_steps <= best_steps:
            continue
        return {"reason": "qualified", "history": prefix[:offset], "check": item["check"],
                "observation": item["after"][0], "preferred_action": item["preferred"],
                "alternative_action": item["alternative"], "horizon_after_check": cap,
                "preferred": {"won": True, "steps_after_check": best_steps},
                "alternative": outcome(state, alt_steps), "revealed": item["revealed"]}
    return {"reason": "no_replayable_outcome_gap" if candidates else "no_grounded_choice"}


def environment(game):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

    fields = textworld.EnvInfos(won=True, admissible_commands=True, policy_commands=True, facts=True)
    return textworld.start(str(game), request_infos=fields, wrappers=[AlfredDemangler(shuffle=False)])


def probe_game(game):
    with redirect_stdout(sys.stderr):
        env = environment(game)
        try:
            result = probe(env)
        finally:
            env.close()
    print(json.dumps(result, ensure_ascii=True), flush=True)


def bounded_worker(command, seconds):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding="utf-8", errors="replace", start_new_session=os.name == "posix")
    try:
        try:
            stdout, stderr = proc.communicate(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"worker {proc.pid} still present after forced stop") from exc
            raise
        if proc.returncode:
            raise RuntimeError(f"worker exited with status {proc.returncode}: {stderr[-2000:].strip()}")
    finally:
        proc.stdout.close()
        proc.stderr.close()
    return stdout


def supervised_probe(game, run_id, deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("B0 reached the two-hour run limit")
    seconds = min(GAME_SECONDS, remaining)
    command = [sys.executable, "-m", "work3.decision_check_preflight",
               "--probe-game", str(game), "--id", run_id]
    try:
        stdout = bounded_worker(command, seconds)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{game.parent.name}: worker exceeded {seconds:.1f}s") from exc
    return json.loads(stdout.splitlines()[-1])


def run(args, directory, progress):
    deadline = monotonic() + RUN_SECONDS
    roots = {name: getattr(args, name).resolve(strict=True) for name in ("train", "seen", "unseen")}
    if len(set(roots.values())) != len(roots) or any(
            a in b.parents for a in roots.values() for b in roots.values() if a != b):
        raise ValueError("train, seen and unseen must be disjoint official split folders")
    versions = {name: importlib.metadata.version(name) for name in (*VERSIONS, "fast-downward-textworld")}
    if any(versions[name] != expected for name, expected in VERSIONS.items()):
        raise ValueError(f"unsupported versions: {versions}, expected {VERSIONS}")
    progress.update("running", detail="decision_check_inputs")
    train = select_games(roots["train"])
    heldout = heldout_seal(train, {key: roots[key] for key in ("seen", "unseen")})
    atomic_json(directory / "heldout_manifest.json", heldout)
    selected, counts = [], Counter()
    for row in train:
        if counts[row["type"]] < PER_TASK:
            selected.append(row)
            counts[row["type"]] += 1
    if not selected:
        raise ValueError("no solvable training games were found")
    started = monotonic()
    totals, reasons = Counter(), Counter()
    progress.update("running", 0, len(selected), detail="decision_check_replay", metrics={"eligible_games": 0})
    with (directory / "decision_checks.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(selected, start=1):
            try:
                result = supervised_probe(roots["train"] / row["file"], args.id, deadline)
            except (Exception, KeyboardInterrupt) as exc:
                raise RuntimeError(f"game {index}/{len(selected)} {row['file']}: {exc}") from exc
            stream.write(json.dumps({**row, **result}, ensure_ascii=False) + "\n")
            stream.flush()
            reasons[result["reason"]] += 1
            if result["reason"] == "qualified":
                totals[row["type"]] += 1
            progress.update("running", index, len(selected), detail="decision_check_replay",
                            metrics={"eligible_games": sum(totals.values())})
    eligible = sum(totals.values())
    passed = eligible >= MIN_ELIGIBLE and sum(value >= MIN_PER_TYPE for value in totals.values()) >= MIN_TYPES
    summary = {"protocol": "decision_check_b0_v1", "repo_revision": progress.commit, "versions": versions,
               "verdict": "GO" if passed else "NO_GO", "tested_games": len(selected),
               "tested_by_type": dict(counts), "eligible_games": eligible, "eligible_by_type": dict(totals),
               "reason_counts": dict(reasons), "heldout": {key: len(rows) for key, rows in heldout.items()},
               "replay_seconds": monotonic() - started}
    atomic_json(directory / "wiki_rsi_decision_check_summary.json", summary)
    progress.update("done" if passed else "blocked", len(selected), len(selected),
                    detail="decision_check_go" if passed else "decision_check_no_go",
                    metrics={"eligible_games": eligible})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def stop_on_signal(signum, _frame):
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--probe-game" and sys.argv[3] == "--id":
        probe_game(Path(sys.argv[2]))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--seen", type=Path, required=True)
    parser.add_argument("--unseen", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(RUN_ID, args.id):
        parser.error("--id must be a fresh decision-check-<name>")
    directory = progress_root() / args.id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    progress = Progress(directory, "decision_check")
    signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        with progress.heartbeat():
            run(args, directory, progress)
    except (Exception, KeyboardInterrupt) as exc:
        row = progress.data["phases"].get("decision_check", {})
        done, total, metrics = row.get("done", 0), row.get("total", 0), row.get("metrics", {})
        atomic_json(directory / "incomplete.json", {
            "verdict": "INCOMPLETE", "repo_revision": progress.commit, "error": str(exc),
            "tested_games": done, "planned_games": total, "eligible_games": metrics.get("eligible_games", 0),
        })
        progress.update("failed", done, total, detail="decision_check_incomplete", metrics=metrics)
        raise


if __name__ == "__main__":
    main()
