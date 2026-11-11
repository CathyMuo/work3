"""run_p0.sh stage wrappers (factory feedback #7, #8): a stage that exits nonzero after its output
was completely written continues; anything short of complete, fresh output still stops."""
import json
import shlex
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from work3 import stage_check
from work3.board import parse_teardown, parse_timings

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p0" / "run_p0.sh"
NGPU = 4
try:
    import pandas as pd  # noqa: F401
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


def helpers() -> str:
    """The helper functions of run_p0.sh, without its top-level setup and stage dispatch."""
    text = SCRIPT.read_text(encoding="utf-8")
    start, end = text.index("\nlog() {"), text.index("\nport_free() {")
    return text[start:end]


def make_ckpt(d: Path, step: int, tracker=None, drop=None):
    actor = d / f"global_step_{step}" / "actor"
    (actor / "huggingface").mkdir(parents=True, exist_ok=True)
    (actor / "huggingface" / "config.json").write_text("{}")
    for kind in ("model", "optim", "extra_state"):
        for r in range(NGPU):
            name = f"{kind}_world_size_{NGPU}_rank_{r}.pt"
            if name != drop:
                (actor / name).write_bytes(b"x")
    (d / f"global_step_{step}" / "data.pt").write_bytes(b"x")
    if tracker is not False:
        (d / "latest_checkpointed_iteration.txt").write_text(str(step if tracker is None else tracker))


# a stage command that writes a complete checkpoint and is then killed while shutting down
WRITE_CKPT_THEN_DIE = (
    "python -c 'import sys; sys.path.insert(0, \"{tests}\"); "
    "from test_run_p0_stages import make_ckpt; from pathlib import Path; "
    "make_ckpt(Path(\"{ckpt}\"), 50{extra})'; kill -9 $$"
)


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
        bindir = Path(self.tmp.name) / "bin"
        bindir.mkdir(exist_ok=True)
        py = bindir / "python"
        if not py.exists():
            py.symlink_to(sys.executable)
        e = dict(os.environ, NGPU=str(NGPU), OUT_ROOT=str(self.out), TAG="work3-test",
                 PYTHONPATH=str(ROOT), PATH=f"{bindir}:{os.environ['PATH']}", **env)
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=e)

    def die_cmd(self, extra=""):
        return "bash -c " + shlex.quote(WRITE_CKPT_THEN_DIE.format(tests=ROOT / "tests", ckpt=self.ckpt, extra=extra))

    def run_ckpt_stage(self, cmd):
        return self.bash(f'checked iter1_challenger ckpt "{self.ckpt}" 50 {NGPU} -- {cmd}; echo AFTER')

    def test_kill_after_complete_checkpoint_continues(self):
        r = self.run_ckpt_stage(self.die_cmd())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("AFTER", r.stdout)
        self.assertIn("WARNING iter1_challenger exited 137", r.stdout)
        lines = (self.out / "timings.txt").read_text().splitlines()
        self.assertEqual([l.split()[2] for l in lines], ["137", "0"])     # the real exit code is kept
        self.assertEqual(lines[0].split()[3], lines[1].split()[3])        # same end time
        self.assertEqual(self.timings()["iter1_challenger"]["rc"], 0)
        w = parse_teardown(self.out / "teardown_warnings.txt")["iter1_challenger"]
        self.assertEqual((w["rc"], w["what"]), (137, "ckpt"))
        self.assertIn("[p0 diag] exit code 137 = signal 9", (self.out / "logs/iter1_challenger.log").read_text())

    def test_incomplete_checkpoint_stops(self):
        for extra in (", tracker=40", ", tracker=False", ", drop=\"optim_world_size_4_rank_3.pt\""):
            with self.subTest(extra=extra):
                shutil.rmtree(self.ckpt, ignore_errors=True)
                r = self.run_ckpt_stage(self.die_cmd(extra))
                self.assertEqual(r.returncode, 137, r.stdout + r.stderr)
                self.assertNotIn("AFTER", r.stdout)
                self.assertIn("FAILED iter1_challenger", r.stdout)
                self.assertIn("[p0 check] ckpt:", r.stdout)

    def test_older_output_never_counts_after_a_failure(self):
        make_ckpt(self.ckpt, 50)                       # complete, but from before this stage started
        old = time.time() - 3600
        os.utime(self.ckpt / "latest_checkpointed_iteration.txt", (old, old))
        r = self.run_ckpt_stage('bash -c "exit 1"')
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("predates this stage", r.stdout)

    def test_success_passes_through(self):
        r = self.bash(f'checked iter1_solver ckpt "{self.ckpt}" 50 {NGPU} -- true; echo AFTER')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("done iter1_solver", r.stdout)
        self.assertFalse((self.out / "teardown_warnings.txt").exists())

    def test_skip_complete_settles_an_earlier_teardown_failure(self):
        make_ckpt(self.ckpt, 50)
        (self.out / "timings.txt").write_text("iter1_challenger 17280 1 1762775654\n")
        skip = f'skip_complete iter1_challenger ckpt "{self.ckpt}" 50 {NGPU} && echo SKIPPED || echo RERUN'
        r = self.bash(skip)
        self.assertIn("SKIPPED", r.stdout, r.stdout + r.stderr)
        t = self.timings()["iter1_challenger"]
        self.assertEqual((t["rc"], t["seconds"], t["end"]), (0, 17280, 1762775654.0))
        self.bash(skip)                                                    # idempotent
        self.assertEqual(len((self.out / "timings.txt").read_text().splitlines()), 2)
        self.assertIn("RERUN", self.bash(skip, FORCE="1").stdout)

    def test_skip_complete_reruns_incomplete(self):
        make_ckpt(self.ckpt, 50, tracker=40)
        r = self.bash(f'skip_complete iter1_challenger ckpt "{self.ckpt}" 50 {NGPU} && echo SKIPPED || echo RERUN')
        self.assertIn("RERUN", r.stdout)
        self.assertIn("no complete output yet", r.stdout)

    def test_hf_convert_kill_after_write_continues(self):
        hf = Path(self.tmp.name) / "solver_iter1_hf"
        write = ("mkdir -p {d} && echo {{}} > {d}/config.json && "
                 "echo '{{\"weight_map\": {{\"a\": \"model-00001-of-00001.safetensors\"}}}}' > {d}/model.safetensors.index.json && "
                 "echo w > {d}/model-00001-of-00001.safetensors && kill -9 $$").format(d=hf)
        r = self.bash(f'checked iter1_convert hf "{hf}" -- bash -c {shlex.quote(write)}; echo AFTER')
        self.assertIn("AFTER", r.stdout, r.stdout + r.stderr)
        (hf / "model-00001-of-00001.safetensors").unlink()
        r = self.bash(f'skip_complete iter1_convert hf "{hf}" && echo SKIPPED || echo RERUN')
        self.assertIn("RERUN", r.stdout)

    def test_background_server_does_not_hold_the_stage(self):
        # upstream scripts start the retriever / reward server with `&` and never stop them;
        # up to r11 those kept the log pipe open and the stage never returned (feedback #9)
        t0 = time.time()
        r = self.bash('timed iter1_gen_data bash -c "echo started; sleep 40 > /dev/null 2>&1 < /dev/null & '
                      'sleep 40 & echo main-done"; echo AFTER')
        self.assertLess(time.time() - t0, 20, r.stdout + r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("AFTER", r.stdout)
        self.assertIn("main-done", (self.out / "logs/iter1_gen_data.log").read_text())
        subprocess.run(["pkill", "-f", "^sleep 40$"])

    def test_one_orchestrator_per_output_dir(self):
        r = self.bash('( take_lock; sleep 4 ) & sleep 1; take_lock; echo GOT')
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn("another run_p0.sh is already running", r.stdout)
        self.assertIn("GOT", self.bash("take_lock; echo GOT").stdout)
        self.assertIn("GOT", self.bash("take_lock; echo GOT", WORK3_P0_LOCK_HELD="1").stdout)

    def test_ordinary_failure_still_stops(self):
        r = self.bash('timed iter1_gen_data bash -c "exit 3"; echo AFTER')
        self.assertEqual(r.returncode, 3)
        self.assertNotIn("AFTER", r.stdout)

    def timings(self):
        return parse_timings(self.out / "timings.txt")


class EffectiveSecondsTest(unittest.TestCase):
    def test_hung_stage_time_is_replaced_by_log_evidence(self):
        from work3.board import effective_seconds
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "logs").mkdir()
            (run / "logs/iter1_challenger.log").write_text(
                "".join(f"step:{i} - critic/score/mean:0.5 - timing_s/step:500.0\n" for i in range(21, 51)))
            row = {"name": "iter1_challenger", "state": "done", "seconds": 66706, "default_hours": 8.0}
            self.assertEqual(effective_seconds(row, run), 50 * 500 + 600)
            row["seconds"] = 30000                       # plausible: kept as recorded
            self.assertEqual(effective_seconds(row, run), 30000)


@unittest.skipUnless(HAVE_PANDAS, "pandas not installed")
class OutputCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pool(self):
        import pandas as pd
        f = self.d / "pool.parquet"
        df = pd.DataFrame({"prompt": ["a", "b"], "data_source": ["hop1", "hop2"],
                           "reward_model": [{"ground_truth": "x"}, {"ground_truth": "y"}]})
        df.to_parquet(f)
        self.assertEqual(stage_check.check_pool(str(f)), [])
        self.assertIn("predates", stage_check.check_pool(str(f), since=time.time() + 60)[0])
        f.write_bytes(f.read_bytes()[:-40])                                  # cut off mid-write
        self.assertIn("unreadable", stage_check.check_pool(str(f))[0])
        df.assign(data_source=[None, "hop2"]).to_parquet(f)
        self.assertIn("empty data_source", stage_check.check_pool(str(f))[0])

    def test_pool_check_must_pass(self):
        f = self.d / "pc.json"
        f.write_text(json.dumps({"ok": True}))
        self.assertEqual(stage_check.check_pool_check(str(f)), [])
        f.write_text(json.dumps({"ok": False}))
        self.assertTrue(stage_check.check_pool_check(str(f)))

    def test_eval(self):
        import pandas as pd
        test = self.d / "test.parquet"
        pd.DataFrame({"data_source": ["searchR1_nq"] * 3 + ["searchR1_bamboogle"] * 2}).to_parquet(test)
        out = self.d / "eval"
        out.mkdir()
        rows = [{"data_source": "searchR1_nq", "score": 1.0}] * 3 + [{"data_source": "searchR1_bamboogle", "score": 0.0}] * 2
        dump = out / "0.jsonl"
        dump.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(stage_check.check_eval(str(out), str(test)), [])
        dump.write_text("".join(json.dumps(r) + "\n" for r in rows[:4]))   # one question missing
        self.assertIn("1 of 2 rows", stage_check.check_eval(str(out), str(test))[0])
        dump.write_text("".join(json.dumps(r) + "\n" for r in rows)[:-15])  # cut off mid-line
        self.assertIn("unreadable", stage_check.check_eval(str(out), str(test))[0])

    def test_cli(self):
        self.assertEqual(stage_check.main(["hf", str(self.d / "none")]), 1)


if __name__ == "__main__":
    unittest.main()
