import json
import tempfile
import unittest
from pathlib import Path

from work3.p1_agent import format_results
from work3.wiki_rsi_progress import Progress, board_runs
from work3.wiki_rsi_replay import paired, raw_passages, score_generation


class WikiReplayTest(unittest.TestCase):
    def test_locator_after_empty_multi_query(self):
        def call(*queries):
            arguments = {"name": "search", "arguments": {"query_list": queries}}
            return f"<tool_call>{json.dumps(arguments)}</tool_call>"

        first = '<tool_response>{"result":"No search results found."}</tool_response>'
        second = format_results([["A Title\nGenuine text."]])
        output = call("A", "B") + first + call("C") + f"<tool_response>{second}</tool_response>"
        self.assertEqual(list(raw_passages(output)), [((3, 1), ("A Title", "Genuine text."))])

    def test_p0_and_strict_scores_are_distinct(self):
        result = score_generation("<answer>Paris</answer> extra", ["Paris"], "stop")
        self.assertEqual((result["p0_em"], result["correct"]), (1, 0))
        self.assertTrue(score_generation("<answer>The Paris</answer>", ["Paris"], "stop")["valid"])
        self.assertFalse(score_generation("</tool_call><answer>Paris</answer>", ["Paris"], "stop")["valid"])
        self.assertFalse(score_generation("<answer>Paris</answer>", ["Paris"], "length")["valid"])

    def test_pairing_uses_question_level_disagreement(self):
        rows = [{"a": {"correct": a}, "b": {"correct": b}} for a, b in [(1, 0), (1, 0), (0, 1), (1, 1)]]
        result = paired(rows, "a", "b")
        self.assertEqual((result["n"], result["wins"], result["losses"], result["gain"]), (4, 2, 1, .25))
        self.assertEqual(result["p_exact"], 1)
        self.assertIsNone(paired([], "a", "b")["gain"])

    def test_board_publishes_only_status_snapshot_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "wiki-test"
            run.mkdir()
            progress = Progress(run, "prepare")
            progress.update("running", 2, 5, detail="checking_history",
                            metrics={"hotpotqa": 4, "private_prompt": 123})
            written = progress.data["updated_at"]
            row = board_runs(root, written + 1201)[0]["phases"]["prepare"]
            self.assertEqual((row["state"], row["done"], row["total"]), ("stalled", 2, 5))
            self.assertEqual(row["metrics"], {"hotpotqa": 4})
            Progress(run, "replay").update("blocked", detail="insufficient_pairs")
            self.assertEqual(set(board_runs(root, written)[0]["phases"]), {"prepare", "replay"})
