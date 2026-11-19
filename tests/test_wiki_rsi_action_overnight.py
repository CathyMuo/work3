import tempfile
import unittest
from pathlib import Path

from work3.wiki_rsi_action_model import scheduled_indices
from work3.wiki_rsi_action_overnight import CHECKPOINTS, JOBS, device_ids, matched_report
from work3.wiki_rsi_action_scale import work_credits
from work3.wiki_rsi_progress import atomic_json


class WikiActionOvernightTest(unittest.TestCase):
    def test_accepts_book_and_song_credits_in_a_matching_source_lead(self):
        self.assertEqual(
            work_credits("The Story", "The Story is a book written by John Stone and illustrated by Mary Cole."),
            ("book", {"author": "John Stone", "illustrator": "Mary Cole"}),
        )
        self.assertEqual(
            work_credits("Blue Song", "Blue Song is a song performed by Mary Cole and composed by John Stone."),
            ("song", {"performer": "Mary Cole", "composer": "John Stone"}),
        )
        self.assertEqual(work_credits("Another", "Blue Song is a song performed by Mary Cole."), (None, {}))

    def test_checkpoint_budget_is_identical_across_seeds(self):
        schedules = [scheduled_indices(155, epochs=8, seed=seed, checkpoints=CHECKPOINTS)
                     for seed in (20261119, 20261120)]
        for schedule in schedules:
            self.assertEqual(sum(map(len, schedule[:78])), 310)
            self.assertEqual(sum(map(len, schedule)), 1240)
        self.assertNotEqual(schedules[0], schedules[1])
        self.assertEqual(device_ids("2,3,0,1"), ["2", "3", "0", "1"])
        with self.assertRaises(ValueError):
            device_ids("0,00,1,2")

    def test_interim_requires_all_four_results_and_matching_token_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            atomic_json(path / "frozen.json", {"prepared_sha256": "fixture", "rows": [], "queries": 0})
            for seed, arm in JOBS[:-1]:
                atomic_json(path / f"{seed}-{arm}-2.json", {"prepared_sha256": "fixture", "seed": seed,
                            "arm": arm, "epoch": 2, "rows": [], "queries": 0,
                            "training": {"steps": 1, "pair_presentations": 2, "unpadded_input_tokens": 10}})
            self.assertIsNone(matched_report(path, [], "fixture", 2))
            seed, arm = JOBS[-1]
            last = {"prepared_sha256": "fixture", "seed": seed, "arm": arm, "epoch": 2,
                    "rows": [], "queries": 0, "training": {"steps": 1, "pair_presentations": 2,
                                                          "unpadded_input_tokens": 11}}
            atomic_json(path / f"{seed}-{arm}-2.json", last)
            with self.assertRaisesRegex(ValueError, "different training budgets"):
                matched_report(path, [], "fixture", 2)
            last["training"]["unpadded_input_tokens"] = 10
            atomic_json(path / f"{seed}-{arm}-2.json", last)
            report = matched_report(path, [], "fixture", 2)
            self.assertEqual(len(report["by_seed"]), 2)
            self.assertEqual(report["epoch"], 2)
