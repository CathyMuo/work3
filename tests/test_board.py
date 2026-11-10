import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from work3 import board as B

CHALLENGER_LOG = (
    "(TaskRunner pid=1) Total training steps: 100\n"
    "Training Progress:  52%|#####     | 52/100 [10:00<09:00, 12.0s/it]\r"
    "Rollout questions with the solver model:  62%|######    | 220/355 [00:45<00:38, 3.72it/s]\r"
    "\U0001F680 Final rewards: Avg 0.41, Max 1.2\n"
    "step:51 - critic/score/mean:0.40 - timing_s/step:300.5 - other:x\n"
    "\U0001F680 Final rewards: Avg 0.45, Max 1.3\n"
    "step:52 - critic/score/mean:0.44 - timing_s/step:280.0\n"
    "WARNING Model attempted to call undefined function: query\n"
    "WARNING Model attempted to call undefined function: query\n"
    "WARNING Model attempted to call undefined function: answer\n"
)


def make_run(root: Path) -> Path:
    run = root / "work3_p0"
    (run / "logs").mkdir(parents=True)
    now = int(time.time())
    for name in ("prepare_train", "prepare_test", "prepare_val", "iter1_challenger", "iter1_gen_data",
                 "iter1_pool_check", "iter1_solver", "iter1_convert"):
        (run / "logs" / f"{name}.log").write_text("ok\n")
        os.utime(run / "logs" / f"{name}.log", (now - 7200, now - 7200))
    (run / "timings.txt").write_text(
        "prepare_train 600\n"                       # p0-r1 format (no exit code)
        f"prepare_test 60 0 {now - 7000}\nprepare_val 5 0 {now - 7000}\n"
        f"iter1_challenger 28800 0 {now - 7000}\niter1_gen_data 14400 0 {now - 7000}\n"
        f"iter1_pool_check 3 0 {now - 7000}\niter1_solver 21600 0 {now - 7000}\n"
        f"iter1_convert 300 0 {now - 7000}\n")
    os.utime(run / "timings.txt", (now - 7000, now - 7000))
    (run / "logs" / "iter2_challenger.log").write_text(CHALLENGER_LOG)
    (run / "pool_check_iter1.json").write_text(json.dumps({"ok": True}))
    (run / "logs" / "retriever_stats.json").write_text(json.dumps({"p50_ms": 12.0, "window_seconds": 300,
                                                                   "updated": now - 30, "p95_ms": 40.0,
                                                                   "requests_per_second": 3.0, "in_flight": 1,
                                                                   "errors_in_window": 0, "total_requests": 9}))
    ev = run / "eval" / "iter1"
    ev.mkdir(parents=True)
    with (ev / "0.jsonl").open("w") as f:
        for d in ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"):
            for i in range(10):
                f.write(json.dumps({"data_source": "searchR1_" + d, "score": 1.0 if i < 3 else 0.0}) + "\n")
    (run / "logs" / "eval1.log").write_text("ok\n")
    os.utime(run / "logs" / "eval1.log", (now - 7200, now - 7200))
    with (run / "timings.txt").open("a") as f:
        f.write(f"eval1 9000 0 {now - 7000}\n")
    os.utime(run / "timings.txt", (now - 7000, now - 7000))
    return run


class ParseTest(unittest.TestCase):
    def test_tqdm_and_steps_and_rewards(self):
        bars = B.tqdm_progress(CHALLENGER_LOG)
        self.assertEqual(bars["Training Progress"]["n"], 52)
        self.assertEqual(bars["Rollout questions with the solver model"]["total"], 355)
        steps = B.step_metrics(CHALLENGER_LOG)
        self.assertEqual([s["step"] for s in steps], [51, 52])
        self.assertAlmostEqual(steps[-1]["critic/score/mean"], 0.44)
        self.assertEqual(B.reward_series(CHALLENGER_LOG), [0.41, 0.45])
        e = B.scan_errors(CHALLENGER_LOG)
        self.assertEqual((e["undefined_function"], e["undefined_function_names"]), (3, {"query": 2, "answer": 1}))

    def test_training_fraction_handles_resumed_proposer(self):
        self.assertAlmostEqual(B.training_fraction("iter2_challenger", {"n": 75, "total": 100}), 0.5)
        self.assertAlmostEqual(B.training_fraction("iter1_solver", {"n": 10, "total": 50}), 0.2)
        self.assertIsNone(B.training_fraction("iter1_solver", None))

    def test_timings_formats(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            p.write_text("a 10\nb 20 1 1700000000\nbad line\nb 30 0 1700000100\n")
            t = B.parse_timings(p)
            self.assertEqual(t["a"], {"seconds": 10, "rc": None, "end": None})
            self.assertEqual(t["b"]["seconds"], 30)

    def test_markdown_escapes_and_renders(self):
        h = B.md_to_html("# T\n\n| a | b |\n| --- | --- |\n| `x` | <s> |\n\n- one\n  cont\n- **two**\n\npara [l](u)")
        self.assertIn("<table>", h); self.assertIn("&lt;s&gt;", h); self.assertIn("<code>x</code>", h)
        self.assertIn("<li>one cont</li>", h); self.assertIn("<b>two</b>", h)
        self.assertNotIn("<s>", h)

    def test_research_reads_repo_files(self):
        r = B.research(B.ROOT)
        self.assertIn("<table>", r["plan_html"])
        self.assertTrue(r["decisions"])
        self.assertIn("frozen_gates", r["state"])


class RunStatusTest(unittest.TestCase):
    def test_stages_current_eta_eval(self):
        with tempfile.TemporaryDirectory() as d:
            run = make_run(Path(d))
            st = B.run_status("p0", "P0", run, B.P0_STAGES, "p0", time.time())
            states = {s["name"]: s["state"] for s in st["stages"]}
            self.assertEqual(states["prepare_train"], "done")
            self.assertEqual(states["iter1_convert"], "done")
            self.assertEqual(states["iter2_challenger"], "running")
            self.assertEqual(states["iter3_solver"], "pending")
            self.assertEqual(st["current"]["stage"], "iter2_challenger")
            self.assertAlmostEqual(st["current"]["fraction"], 0.04)
            self.assertGreater(st["eta_seconds"], 0)
            self.assertAlmostEqual(st["evals"]["1"]["average"], 0.3)
            labels = [l["label"] for l in st["lanes"]]
            self.assertEqual(labels, ["准备", "第 1 轮", "第 2 轮", "第 3 轮", "收尾"])
            lane1 = {s["label"]: s for s in st["lanes"][1]["segments"]}
            self.assertEqual([s["state"] for s in st["lanes"][1]["segments"]], ["done"] * 4)
            self.assertAlmostEqual(lane1["评测"]["em"], 0.3)
            seg2 = st["lanes"][2]["segments"][0]
            self.assertEqual((seg2["label"], seg2["state"]), ("出题者训练", "running"))
            self.assertAlmostEqual(seg2["fraction"], 0.04)
            self.assertTrue(0 < st["overall_fraction"] < 1)
            self.assertGreater(st["finish_at"], time.time())
            self.assertIn("出题者训练", st["current"]["explain"])
            # per-stage and per-iteration time
            self.assertGreater(st["current"]["stage_remaining_seconds"], 0)
            self.assertGreater(st["current"]["stage_finish_at"], time.time())
            lane1, lane2 = st["lanes"][1], st["lanes"][2]
            self.assertEqual(lane1["remaining_seconds"], 0)
            self.assertGreater(lane1["done_seconds"], 0)
            self.assertGreater(lane2["remaining_seconds"], 0)
            self.assertAlmostEqual(lane2["remaining_seconds"] + lane2["done_seconds"], lane2["est_seconds"], delta=2)
            self.assertGreater(st["est_total_seconds"], st["eta_seconds"])
            self.assertEqual(st["retriever"]["p50_ms"], 12.0)
            self.assertEqual(st["pool_checks"], {"pool_check_iter1": True})

    def test_failed_and_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "work3_p0"; (run / "logs").mkdir(parents=True)
            now = int(time.time())
            (run / "logs" / "prepare_train.log").write_text("Traceback (most recent call last):\n x\n")
            os.utime(run / "logs" / "prepare_train.log", (now - 100, now - 100))
            (run / "timings.txt").write_text("prepare_train 5\n")
            os.utime(run / "timings.txt", (now - 90, now - 90))
            st = B.run_status("p0", "P0", run, B.P0_STAGES, "p0", now)
            self.assertEqual(st["stages"][0]["state"], "failed")
            (run / "logs" / "prepare_train.log").write_text("rerunning\n")      # newer than timings
            st = B.run_status("p0", "P0", run, B.P0_STAGES, "p0", time.time())
            self.assertEqual(st["stages"][0]["state"], "running")


class MetricsTest(unittest.TestCase):
    def test_incremental_parse_and_all_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.log"
            p.write_text("(TaskRunner pid=3) step:1 - critic/score/mean:0.1 - actor/entropy:nan - "
                         "val-core/searchR1_nq/reward/mean@1:0.30 - name:abc\n\U0001F680 Final rewards: Avg 0.2, Max 1\n"
                         "step:2 - critic/score/mean:0.")
            c = B.MetricsCache()
            first = c.get(p)
            self.assertEqual([r["step"] for r in first["rows"]], [1])
            self.assertNotIn("actor/entropy", first["rows"][0])          # NaN dropped
            self.assertIn("val-core/searchR1_nq/reward/mean@1", first["keys"])
            self.assertEqual(first["rewards"], [0.2])
            with p.open("a") as f:                                         # finish the partial line
                f.write("25 - timing_s/step:12\r\n")
            second = c.get(p)
            self.assertEqual(second["rows"][1], {"step": 2, "critic/score/mean": 0.25, "timing_s/step": 12.0})
            p.write_text("step:9 - a/b:1\n")                               # truncated / rewritten file
            self.assertEqual([r["step"] for r in c.get(p)["rows"]], [9])

    def test_reward_log_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reward_log.jsonl"
            p.write_text(json.dumps({"call": 1, "batch": 4, "valid": 2, "k_hist": {"0": 1, "1": 1},
                                     "mean_difficulty": 0.5, "mean_w_mixed": 0.8, "calls": {"completion_tokens": 9},
                                     "seconds": 3}) + "\nnot json\n")
            rows = B.reward_log_rows(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]["valid_share"], rows[0]["k=1 share"], rows[0]["completion_tokens"]),
                             (0.5, 0.5, 9))

    def test_board_metrics_compare_and_history(self):
        with tempfile.TemporaryDirectory() as d:
            run = make_run(Path(d))
            (run / "logs" / "iter1_challenger.log").write_text(
                "step:1 - critic/score/mean:0.1\nstep:2 - critic/score/mean:0.2\n")
            board = B.Board(drzero_dir=d, arm_root=str(Path(d) / "none"))
            st = board.metric_stages("p0")
            self.assertEqual([s["name"] for s in st["stages"]][:2], ["iter1_challenger", "iter1_solver"])
            m = board.metrics("p0", "iter2_challenger", compare=True)
            self.assertTrue(m["compare"])
            self.assertEqual(set(m["series"]), {"iter1_challenger", "iter2_challenger"})
            self.assertEqual(m["series"]["iter2_challenger"]["step_offset"], 50)
            self.assertIsNone(board.metrics("p0", "../../etc/passwd"))
            h = B.History(board, gpu_fn=lambda: [{"util": 50.0, "power": 300.0, "mem_used": 1024.0}])
            h.sample_once(now=100.0); h.sample_once(now=110.0)
            snap = h.snapshot()
            self.assertEqual(len(snap["gpu"]), 2)
            self.assertEqual(len(snap["retriever"]["p0"]), 1)          # same stats file sampled once


class ServerTest(unittest.TestCase):
    def test_http_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(Path(d))
            board = B.Board(drzero_dir=d, arm_root=str(Path(d) / "none"))
            from http.server import ThreadingHTTPServer
            srv = ThreadingHTTPServer(("127.0.0.1", 0), B.make_handler(board))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                page = urllib.request.urlopen(base + "/").read().decode()
                self.assertIn("/api/state", page)
                st = json.loads(urllib.request.urlopen(base + "/api/state").read())
                self.assertEqual(st["runs"][0]["current"]["stage"], "iter2_challenger")
                self.assertFalse(st["runs"][1]["exists"])
                log = json.loads(urllib.request.urlopen(base + "/api/log?run=p0").read())
                self.assertEqual(log["file"], "iter2_challenger.log")
                res = json.loads(urllib.request.urlopen(base + "/api/research").read())
                self.assertIn("plan_html", res)
                m = json.loads(urllib.request.urlopen(base + "/api/metrics?run=p0&stage=iter2_challenger").read())
                self.assertEqual([r["step"] for r in m["series"]["iter2_challenger"]["rows"]], [51, 52])
                self.assertIn("stages", json.loads(urllib.request.urlopen(base + "/api/stages?run=p0").read()))
                self.assertIn("gpu", json.loads(urllib.request.urlopen(base + "/api/history").read()))
                self.assertEqual(json.loads(urllib.request.urlopen(base + "/api/rewardlog?run=p0").read())["rows"], [])
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/api/metrics?run=p0&stage=nope")
                brief = urllib.request.urlopen(base + "/api/briefing").read().decode()
                self.assertIn("NCP", brief)
                rm = st["roadmap"]
                self.assertEqual([p["key"] for p in rm], ["P0", "P1", "P2", "M", "S"])
                self.assertEqual(rm[0]["state"], "running")
                self.assertEqual(rm[0]["result"], "第 1 轮 EM 0.300")
                self.assertEqual(rm[1]["state"], "pending")
                # phase-level time: P0 running, P1 from defaults, P2 arms from P0's observed stage times
                self.assertGreater(rm[0]["remaining_seconds"], 0)
                self.assertEqual(rm[1]["est_seconds"], round(4.5 * 3600))
                self.assertEqual(rm[1]["remaining_seconds"], rm[1]["est_seconds"])
                self.assertGreater(rm[2]["est_seconds"], 2 * 3600 * 20)
                self.assertIsNone(rm[3]["est_seconds"])
                self.assertEqual(st["pilots_remaining_seconds"],
                                 sum(p["remaining_seconds"] for p in rm[:3]))
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/api/log?run=nope")
            finally:
                srv.shutdown()


if __name__ == "__main__":
    unittest.main()
