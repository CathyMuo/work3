"""run_p0.sh stage wrappers: a training stage that exits nonzero after its complete final
checkpoint (factory feedback #7) continues; anything short of a complete checkpoint still stops."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from work3.board import parse_timings

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p0" / "run_p0.sh"
NGPU = 4


def helpers() -> str:
    """The helper functions of run_p0.sh, without its top-level setup and stage dispatch."""
    text = SCRIPT.read_text(encoding="utf-8")
    start, end = text.index("\nlog() {"), text.index("\nport_free() {")
    return text[start:end]


def make_ckpt(d: Path, step: int, tracker=None, drop=None):
    actor = d / f"global_step_{step}" / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "huggingface" / "config.json").write_text("{}")
    for kind in ("model", "optim", "extra_state"):
        for r in range(NGPU):
            name = f"{kind}_world_size_{NGPU}_rank_{r}.pt"
            if name != drop:
                (actor / name).write_bytes(b"x")
    (d / f"global_step_{step}" / "data.pt").write_bytes(b"x")
    if tracker is not False:
        (d / "latest_checkpointed_iteration.txt").write_text(str(step if tracker is None else tracker))


class StageWrapperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "out"
        (self.out / "logs").mkdir(parents=True)
        self.ckpt = Path(self.tmp.name) / "challenger_iter1"

    def tearDown(self):
        self.tmp.cleanup()

    def bash(self, body: str, **env):
        script = "set -euo pipefail\n" + helpers() + "\ncleanup_servers() { :; }\n" + body
        e = dict(os.environ, NGPU=str(NGPU), OUT_ROOT=str(self.out), TAG="work3-test", **env)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e)

    def timings(self):
        return parse_timings(self.out / "timings.txt")

    def test_teardown_exit_after_complete_checkpoint_continues(self):
        make_ckpt(self.ckpt, 50)
        r = self.bash(f'train_stage iter1_challenger "{self.ckpt}" 50 bash -c "exit 1"; echo AFTER')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("AFTER", r.stdout)
        self.assertIn("WARNING iter1_challenger exited 1", r.stdout)
        self.assertEqual(self.timings()["iter1_challenger"]["rc"], 0)
        lines = (self.out / "timings.txt").read_text().splitlines()
        self.assertEqual([l.split()[2] for l in lines], ["1", "0"])        # the real exit code is kept
        self.assertEqual(lines[0].split()[3], lines[1].split()[3])         # same end time
        self.assertIn("iter1_challenger 1 global_step_50", (self.out / "teardown_warnings.txt").read_text())

    def test_exit_with_incomplete_checkpoint_stops(self):
        for kw in ({"tracker": 40}, {"tracker": False}, {"drop": "optim_world_size_4_rank_3.pt"}):
            with self.subTest(**kw):
                if self.ckpt.exists():
                    import shutil
                    shutil.rmtree(self.ckpt)
                make_ckpt(self.ckpt, 50, **kw)
                r = self.bash(f'train_stage iter1_challenger "{self.ckpt}" 50 bash -c "exit 1"; echo AFTER')
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertNotIn("AFTER", r.stdout)
                self.assertIn("FAILED iter1_challenger", r.stdout)
                self.assertIn("[p0 ckpt]", r.stdout)

    def test_success_passes_through(self):
        r = self.bash(f'train_stage iter1_solver "{self.ckpt}" 50 true; echo AFTER')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("done iter1_solver", r.stdout)
        self.assertFalse((self.out / "teardown_warnings.txt").exists())

    def test_skip_trained_settles_an_earlier_teardown_failure(self):
        make_ckpt(self.ckpt, 50)
        (self.out / "timings.txt").write_text("iter1_challenger 17280 1 1762775654\n")
        r = self.bash(f'skip_trained iter1_challenger "{self.ckpt}" 50 && echo SKIPPED')
        self.assertIn("SKIPPED", r.stdout, r.stdout + r.stderr)
        t = self.timings()["iter1_challenger"]
        self.assertEqual((t["rc"], t["seconds"], t["end"]), (0, 17280, 1762775654.0))
        r = self.bash(f'skip_trained iter1_challenger "{self.ckpt}" 50 && echo SKIPPED')   # idempotent
        self.assertEqual(len((self.out / "timings.txt").read_text().splitlines()), 2)

    def test_skip_trained_reruns_incomplete_or_forced(self):
        make_ckpt(self.ckpt, 50, tracker=40)
        r = self.bash(f'skip_trained iter1_challenger "{self.ckpt}" 50 && echo SKIPPED || echo RERUN')
        self.assertIn("RERUN", r.stdout)
        self.assertIn("exists but is incomplete", r.stdout)
        (self.ckpt / "latest_checkpointed_iteration.txt").write_text("50")
        r = self.bash(f'skip_trained iter1_challenger "{self.ckpt}" 50 && echo SKIPPED || echo RERUN', FORCE="1")
        self.assertIn("RERUN", r.stdout)
        r = self.bash(f'skip_trained iter1_challenger "{self.ckpt}/none" 50 && echo SKIPPED || echo RERUN')
        self.assertIn("RERUN", r.stdout)

    def test_ordinary_stage_failure_still_stops(self):
        r = self.bash('timed iter1_gen_data bash -c "exit 3"; echo AFTER')
        self.assertEqual(r.returncode, 3)
        self.assertNotIn("AFTER", r.stdout)


if __name__ == "__main__":
    unittest.main()
