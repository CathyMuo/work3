import unittest

from work3 import speed_probe as S

METRICS = """# HELP sglang:num_running_reqs x
sglang:num_running_reqs{model_name="m",tp_rank="0"} 12.0
sglang:num_running_reqs{model_name="m",tp_rank="1"} 3.0
sglang:num_queue_reqs{model_name="m",tp_rank="0"} 5.0
sglang:token_usage{model_name="m",tp_rank="0"} 0.97
sglang:token_usage{model_name="m",tp_rank="1"} 0.40
sglang:gen_throughput{model_name="m",tp_rank="0"} 1500.5
"""


class ProbeTest(unittest.TestCase):
    def test_parse_prometheus(self):
        m = S.parse_prometheus(METRICS)
        self.assertEqual(m["sglang:num_running_reqs"], [12.0, 3.0])
        self.assertEqual(m["sglang:token_usage"], [0.97, 0.40])

    def sample(self, utils, reward=None):
        return {"t": 0, "gpus": [{"index": str(i), "uuid": f"u{i}", "util": u, "mem": 1000, "power": 100}
                                 for i, u in enumerate(utils)], "reward": reward}

    def test_idle_gpu_and_kv_bound_verdict(self):
        full = {"running": 120, "queued": 30, "token_usage_max": 0.99, "gen_throughput": 900, "ranks": 2}
        samples = [self.sample([90, 80, 0, 0], full) for _ in range(5)]
        procs = [{"pid": 1, "gpu": "0", "mem_mb": 1, "kind": "reward-sglang"},
                 {"pid": 2, "gpu": "2", "mem_mb": 1, "kind": "reward-sglang"}]
        res = S.summarize(samples, {"tp_size": 2, "dp_size": 2, "mem_fraction_static": 0.25,
                                    "max_total_num_tokens": 400000}, procs, {"p50_ms": 20, "p95_ms": 90})
        self.assertEqual(res["per_gpu"]["2"]["busy_share"], 0)
        text = "\n".join(res["verdict"])
        self.assertIn("GPU 2,3", text)
        self.assertIn("DP 分发可能没有生效", text)
        self.assertIn("KV 缓存经常占满", text)
        self.assertIn("tp=2，dp=2", text)

    def test_latency_bound_and_no_metrics(self):
        idle = {"running": 6, "queued": 0, "token_usage_max": 0.2, "gen_throughput": 300, "ranks": 4}
        res = S.summarize([self.sample([5, 5, 5, 5], idle)], None, [], None)
        self.assertIn("长尾", "\n".join(res["verdict"]))
        self.assertIn("分批依次跑只会把各批的尾巴相加", "\n".join(res["verdict"]))
        res = S.summarize([self.sample([5, 5, 5, 5])], None, [], None)
        self.assertIn("没有开启 /metrics", "\n".join(res["verdict"]))

    def test_retriever_queueing_is_not_called_slow(self):
        retr = {"p50_ms": 405, "p95_ms": 1680, "p50_ms_low_load": 25, "p95_ms_low_load": 60}
        text = "\n".join(S.summarize([self.sample([5, 5, 5, 5])], None, [], retr)["verdict"])
        self.assertIn("来自请求排队", text)
        self.assertNotIn("检索本身可能是瓶颈", text)
        slow = dict(retr, p95_ms_low_load=800)
        self.assertIn("检索本身可能是瓶颈", "\n".join(S.summarize([self.sample([5])], None, [], slow)["verdict"]))
        old = {"p50_ms": 405, "p95_ms": 1680}
        self.assertIn("不能据此判断", "\n".join(S.summarize([self.sample([5])], None, [], old)["verdict"]))

    def test_unreachable_server(self):
        self.assertIsNone(S.reward_metrics(1))
        self.assertIsNone(S.server_info(1))


if __name__ == "__main__":
    unittest.main()
