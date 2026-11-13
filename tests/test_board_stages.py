"""Board: progress and ETA for every stage kind, robust step parsing, the ablation run."""
import json
import tempfile
import time
import unittest
from pathlib import Path

from work3 import board as B


def run_dir(tmp: str) -> Path:
    d = Path(tmp) / "work3_p0"
    (d / "logs").mkdir(parents=True)
    return d


class StepParseTest(unittest.TestCase):
    def test_numpy2_tensor_and_ansi(self):
        self.assertEqual(B._num("np.float64(0.25)"), 0.25)
        self.assertEqual(B._num("np.int64(7)"), 7.0)
        self.assertEqual(B._num("tensor(0.5, device='cuda:0')"), 0.5)
        self.assertIsNone(B._num("nan"))
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "x.log"
            log.write_text("\x1b[36m(TaskRunner pid=1)\x1b[0m step:1 - critic/score/mean:np.float64(0.3) - "
                           "timing_s/step:500.0\n(TaskRunner pid=1) step:2 - critic/score/mean:0.4\n")
            d = B.MetricsCache().get(log)
            self.assertEqual([r["critic/score/mean"] for r in d["rows"]], [0.3, 0.4])
            self.assertEqual(d["unparsed_step_lines"], 0)
            self.assertEqual(B.step_metrics(log.read_text())[0]["critic/score/mean"], 0.3)

    def test_stage_types(self):
        self.assertEqual([B.stage_type(n) for n in ("ablate_micro2", "ablate_micro2_eval", "ablate_micro1_convert",
                                                  "iter2_solver", "eval0", "pool3")],
                         ["solver", "eval", "convert", "solver", "eval", "pool"])
        self.assertEqual([B.stage_kind(n) for n in ("iter1_challenger", "ablate_micro1", "eval3", "ablate_micro2_eval",
                                                  "iter1_gen_data", "pool1", "iter1_convert")],
                         ["train", "train", "eval", "eval", "gen", "p1pool", "other"])


class ProgressTest(unittest.TestCase):
    def status(self, d, plan, kind="p0"):
        return B.run_status("x", "X", d, plan, kind, time.time())

    def test_eval_progress_from_validation_batches(self):
        with tempfile.TemporaryDirectory() as t:
            d = run_dir(t)
            (d / "logs" / "eval1.log").write_text("validation generation end\n" * 50)
            (d / "logs" / "eval1.start").write_text(str(int(time.time() - 1800)))
            s = self.status(d, [("eval1", "评测", 3.0)])
            c = s["current"]
            self.assertAlmostEqual(c["fraction"], 50 / 203)
            self.assertIn("12,800 / 51,713", c["progress_text"])
            self.assertAlmostEqual(c["stage_remaining_seconds"], 1800 * (203 - 50) / 50, delta=60)

    def test_gen_bar_skips_loading_bars(self):
        bars = B.tqdm_progress("Loading safetensors checkpoint shards: 100%|##| 2/2 [00:01<00:00]\n"
                               "Capturing batches (bs=160 avail_mem=1 GB): 100%|##| 35/35 [00:10<00:00]\n"
                               "(main_task pid=1) 30%|###  | 20/67 [1:04:00<2:30:00, 192.96s/it]\n")
        b = B.gen_bar(bars)
        self.assertEqual((b["n"], b["total"]), (20, 67))

    def test_p1_pool_progress_and_k(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "work3_p1"
            (d / "logs").mkdir(parents=True)
            (d / "logs" / "pool1.log").write_text("[p1 pool1] batch 1: 500 drawn, 500 to run\n")
            (d / "logs" / "pool1.start").write_text(str(int(time.time() - 600)))
            with (d / "pool1_questions.jsonl").open("w") as f:
                for q in range(100):
                    f.write(json.dumps({"qid": q, "scores": [1, 0, 0, 0, 0], "k": 1 if q % 2 else 3}) + "\n")
            s = self.status(d, [("pool1", "题池 1", 2.0)], kind="p1")
            p = s["current"]["p1"]
            self.assertEqual((p["questions"], p["drawn"], p["k_hist"]["1"], p["k_hist"]["3"]), (100, 500, 50, 50))
            self.assertAlmostEqual(s["current"]["fraction"], 100 / (500 + 125))
            self.assertIsNotNone(s["current"]["stage_remaining_seconds"])
            self.assertIn("pool1", s["p1_pools"])

    def test_no_progress_output_uses_start_time(self):
        with tempfile.TemporaryDirectory() as t:
            d = run_dir(t)
            (d / "logs" / "iter1_convert.log").write_text("working\n")
            (d / "logs" / "iter1_convert.start").write_text(str(int(time.time() - 60)))
            c = self.status(d, [("iter1_convert", "转 HF", 0.1)])["current"]
            self.assertAlmostEqual(c["fraction"], 60 / 360, delta=0.02)
            self.assertIn("典型耗时", c["eta_basis"])

    def test_rate_from_board_samples_without_start(self):
        key = ("k", "s")
        self.assertIsNone(B.progress_rate(key, 1000.0, 0.10))
        self.assertIsNone(B.progress_rate(key, 1060.0, 0.12))          # under 2 minutes
        self.assertAlmostEqual(B.progress_rate(key, 1300.0, 0.20), 0.10 / 300)


class AblationRunTest(unittest.TestCase):
    def test_ablation_spec_lanes_and_compare(self):
        with tempfile.TemporaryDirectory() as t:
            dz = Path(t) / "drzero"
            d = dz / "work3_p0"
            (d / "logs").mkdir(parents=True)
            line = "(TaskRunner pid=1) step:{} - critic/score/mean:np.float64(0.5) - timing_s/step:200.0\n"
            (d / "logs" / "iter3_solver.log").write_text("".join(line.format(s) for s in range(1, 51)))
            (d / "logs" / "ablate_micro2.log").write_text("".join(line.format(s) for s in range(1, 11)) +
                                                          "Training Progress:  20%|##  | 10/50 [00:33:20<2:13:20, 200.00s/it]\n")
            board = B.Board(str(dz), None)
            st = board.state()
            ab = next(r for r in st["runs"] if r["id"] == "p0-ablate")
            self.assertEqual(ab["current"]["stage"], "ablate_micro2")
            self.assertEqual([l["label"] for l in ab["lanes"]], ["微批次 2", "微批次 1"])
            self.assertIn("200 秒/步", ab["current"]["eta_basis"])
            stages = board.metric_stages("p0-ablate")
            self.assertEqual([(s["name"], s["steps"]) for s in stages["stages"]], [("ablate_micro2", 10)])
            m = board.metrics("p0-ablate", "ablate_micro2", compare=True)
            self.assertEqual(sorted(m["series"]), ["ablate_micro2", "iter3_solver"])
            p0 = next(r for r in st["runs"] if r["id"] == "p0")
            self.assertNotIn("ablate", (p0["current"] or {}).get("stage", ""))   # separate runs
            self.assertNotIn("ablate_micro2", [s["name"] for s in board.metric_stages("p0")["stages"]])


if __name__ == "__main__":
    unittest.main()
