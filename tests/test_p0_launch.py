import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from work3 import p0_launch, val_subset


class LaunchTest(unittest.TestCase):
    def test_parse_map_requires_existing_files(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "test.parquet"
            f.write_text("x")
            self.assertEqual(p0_launch.parse_map([f"test.parquet={f}"]),
                             {"test.parquet": str(f.resolve())})
            with self.assertRaises(FileNotFoundError):
                p0_launch.parse_map([f"train.parquet={d}/missing.parquet"])
            with self.assertRaises(ValueError):
                p0_launch.parse_map(["no-equals-sign"])

    def test_offline_download_returns_mapped_path_only(self):
        dl = p0_launch.make_offline_download({"test.parquet": "/data/t.parquet"})
        self.assertEqual(dl(repo_id="r", filename="test.parquet", repo_type="dataset",
                            local_dir="/tmp"), "/data/t.parquet")
        with self.assertRaises(FileNotFoundError):
            dl(repo_id="r", filename="train.parquet")

    def test_offline_hf_seen_by_from_import_in_target(self):
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "train.parquet"
            local.write_text("x")
            out = Path(d) / "out.txt"
            script = Path(d) / "target.py"
            script.write_text(
                "import sys\nfrom huggingface_hub import hf_hub_download\n"
                "p = hf_hub_download(repo_id='r', filename='train.parquet')\n"
                "open(sys.argv[1], 'w').write(p)\n")
            fake = types.ModuleType("huggingface_hub")
            saved = sys.modules.get("huggingface_hub"), list(sys.argv)
            sys.modules["huggingface_hub"] = fake
            try:
                p0_launch.main(["offline-hf", "--map", f"train.parquet={local}", "--",
                                str(script), str(out)])
            finally:
                if saved[0] is None:
                    sys.modules.pop("huggingface_hub", None)
                else:
                    sys.modules["huggingface_hub"] = saved[0]
                sys.argv = saved[1]
            self.assertEqual(out.read_text(), str(local.resolve()))

    def test_serve_forces_uvicorn_port(self):
        calls = []
        fake = types.ModuleType("uvicorn")
        fake.run = lambda app, **kw: calls.append(kw)
        saved = sys.modules.get("uvicorn"), list(sys.argv)
        sys.modules["uvicorn"] = fake
        try:
            with tempfile.TemporaryDirectory() as d:
                script = Path(d) / "server.py"
                script.write_text("import uvicorn\nuvicorn.run(object(), host='0.0.0.0', port=8000)\n")
                p0_launch.main(["serve", "--port", "8123", "--", str(script)])
        finally:
            if saved[0] is None:
                sys.modules.pop("uvicorn", None)
            else:
                sys.modules["uvicorn"] = saved[0]
            sys.argv = saved[1]
        self.assertEqual(calls, [{"host": "0.0.0.0", "port": 8123}])

    def test_latency_stats_separates_queueing(self):
        stats = p0_launch.LatencyStats(app=None, path=None)
        for i in range(10):                                   # quiet: fast service
            stats.in_flight += 1; stats.record(100.0 + i, 0.025, 200, ahead=0)
        for i in range(10):                                   # burst: long waits behind 50 others
            stats.in_flight += 1; stats.record(120.0 + i, 0.9, 200, ahead=50)
        s = stats.snapshot(130.0)
        self.assertEqual(s["low_load_requests"], 10)
        self.assertEqual(s["p95_ms_low_load"], 25.0)
        self.assertGreater(s["p95_ms"], 800)
        self.assertEqual(s["mean_in_flight_at_arrival"], 25.0)

    def test_missing_separator_exits(self):
        with self.assertRaises(SystemExit):
            p0_launch.main(["serve", "--port", "1"])


class ValSubsetTest(unittest.TestCase):
    def test_allocation_sums_and_is_proportional(self):
        counts = {"popqa": 14267, "2wikimultihopqa": 12576, "triviaqa": 11313,
                  "hotpotqa": 7405, "nq": 3610, "musique": 2417, "bamboogle": 125}
        a = val_subset.allocate(counts)
        self.assertEqual(sum(a.values()), 1200)
        for k, v in counts.items():
            self.assertLessEqual(abs(a[k] - 1200 * v / 51713), 1)
        self.assertEqual(a["bamboogle"], 3)

    def test_allocation_too_large_raises(self):
        with self.assertRaises(ValueError):
            val_subset.allocate({"a": 3}, size=4)

    def test_subset_is_deterministic_and_stratified(self):
        import pandas as pd
        df = pd.DataFrame({"data_source": ["a"] * 300 + ["b"] * 100,
                           "x": range(400)})
        s1 = val_subset.subset(df, size=40)
        s2 = val_subset.subset(df, size=40)
        self.assertEqual(s1["x"].tolist(), s2["x"].tolist())
        self.assertEqual(s1["data_source"].value_counts().to_dict(), {"a": 30, "b": 10})
        self.assertEqual(len(set(s1["x"])), 40)


if __name__ == "__main__":
    unittest.main()
