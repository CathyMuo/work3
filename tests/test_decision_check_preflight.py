import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from work3.decision_check_archive import archive
from work3.decision_check_preflight import bounded_worker, probe


OPEN = "open cabinet 1"
GOOD = "take apple 1 from cabinet 1"
ALT = "take mug 1 from cabinet 1"
DROP = "put mug 1 in cabinet 1"
DONE = "put apple 1 in bowl 1"


def state(feedback, plan=(), legal=None, won=False):
    return {"feedback": feedback, "admissible_commands": plan if legal is None else legal,
            "facts": (feedback,), "won": won, "policy_commands": plan}


class ToyGame:
    def __init__(self, alternate_observations=False):
        self.resets = 0
        self.alternate_observations = alternate_observations

    def reset(self):
        self.resets += 1
        self.moves = []
        text = "You see cabinet 1."
        if self.alternate_observations and self.resets > 1:
            text += " The room changed."
        return state(text, [OPEN])

    def step(self, action):
        self.moves.append(action)
        result = {
            OPEN: state("The cabinet holds apple 1 and mug 1.", [GOOD], [GOOD, ALT]),
            GOOD: state("You take apple 1.", [DONE]),
            ALT: state("You take mug 1.", [DROP]),
            DROP: state("You put mug 1 down.", [GOOD]),
            DONE: state("You won.", won=True),
        }[action]
        return result, int(result["won"]), result["won"]


class DecisionCheckTests(unittest.TestCase):
    def test_actual_branch_cost_and_replay(self):
        game = ToyGame()
        result = probe(game)
        self.assertEqual(result["reason"], "qualified")
        self.assertEqual(result["preferred"], {"won": True, "steps_after_check": 2})
        self.assertEqual(result["alternative"], {"won": True, "steps_after_check": 4})
        self.assertEqual(game.moves, [OPEN, ALT, DROP, GOOD, DONE])

    def test_replay_drift_does_not_qualify(self):
        result = probe(ToyGame(alternate_observations=True))
        self.assertEqual(result, {"reason": "no_replayable_outcome_gap"})

    def test_worker_times_out_even_when_it_never_returns(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            bounded_worker([sys.executable, "-c", "import time; time.sleep(30)"], 0.15)
        self.assertEqual(bounded_worker([sys.executable, "-c", "print('ready')"], 5).strip(), "ready")

    def test_archive_preserves_partial_count_and_refuses_a_live_run(self):
        counts = {
            "pick_and_place_simple": (40, 2), "pick_clean_then_place_in_recep": (40, 3),
            "pick_heat_then_place_in_recep": (40, 3), "pick_cool_then_place_in_recep": (40, 1),
            "look_at_obj_in_light": (40, 0), "pick_two_obj_and_place": (21, 0),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "decision-check-b0-test"
            run.mkdir()
            proc = root / "proc"
            proc.mkdir()
            live = proc / "142"
            live.mkdir()
            cmdline = live / "cmdline"
            cmdline.write_bytes(b"python\0-m\0work3.decision_check_preflight\0--id\0decision-check-b0-test\0")
            rows = [{"type": task, "file": f"{task}/{index}", "sha256": f"{task}/{index}",
                     "reason": "qualified" if index < eligible else "no_grounded_choice"}
                    for task, (checked, eligible) in counts.items() for index in range(checked)]
            records = run / "decision_checks.jsonl"
            raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
            records.write_bytes(raw)
            progress = run / "progress.json"
            snapshot = {"state": "running", "done": 221, "total": 240, "updated_at": 100, "started_at": 80,
                        "heartbeat_at": 110, "commit": "cb243f3", "metrics": {"eligible_games": 9}}
            progress.write_text(json.dumps({"schema_version": 1, "updated_at": 100,
                                            "phases": {"decision_check": snapshot}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "142"):
                archive(run, 500, proc)
            cmdline.write_bytes(b"python\0-m\0unrelated_training\0")
            result = archive(run, 500, proc)
            self.assertEqual(result["verdict"], "INCOMPLETE")
            self.assertEqual(result["research_gate"], "UNREACHABLE")
            self.assertEqual(result["eligible_upper_bound"], 28)
            self.assertEqual(result["eligible_type_upper_bound"], 1)
            self.assertEqual(result["original_progress"], snapshot)
            self.assertEqual(archive(run, 900, proc), result)
            self.assertEqual(records.read_bytes(), raw)
            self.assertEqual(json.loads(progress.read_text(encoding="utf-8"))["phases"]["decision_check"]["done"], 221)
            self.assertFalse((run / "wiki_rsi_decision_check_summary.json").exists())
