import unittest

from work3 import wiki_rsi_gap_follow_up as gap
from work3 import wiki_rsi_switch_transfer as transfer


class GapFollowUpTest(unittest.TestCase):
    def test_same_format_and_fallback_for_both_conditions(self):
        result = {"text": "Focus: Place and opening date\nQuery: Town hall opened date", "finish": "stop"}
        self.assertEqual(gap.follow_up(result, "town history", "old rule"), ("Town hall opened date", "model"))
        self.assertEqual(gap.follow_up(result, "Town hall opened date!", "old rule"), ("old rule", "rule_fallback"))
        self.assertEqual(gap.follow_up({**result, "finish": "length"}, "old rule", "old rule"), (None, "no_second"))

    def test_stop_is_retained_in_the_hindsight_upper_bound(self):
        info = {"correct": 1, "evidence_complete": 0, "queries": 1}
        miss = {"correct": 0, "evidence_complete": 1, "queries": 2}
        examples = [{"dataset": dataset, "first_only": info, "frozen_follow_up": miss, "outcomes": [miss]}
                    for dataset in transfer.SUPPORT_DATASETS]
        values = {f"{seed}-{arm}": [miss, miss] for seed in transfer.parallel.SEEDS
                  for arm in ("ordinary", "interaction")}
        report = transfer.transfer_report(examples, values)
        for stats in report.values():
            self.assertEqual(stats["oracle_answer_count"], 0)
            self.assertEqual(stats["oracle_with_stop_answer_count"], 1)
            self.assertEqual(stats["oracle_with_stop_evidence_count"], 1)
        self.assertEqual(transfer.research_verdict(report), "no_go")

    def test_research_verdict_requires_replicated_answer_improvements(self):
        by_seed = {seed: {"correct": {"gain": 1 / 192}} for seed in transfer.parallel.SEEDS}
        groups = {dataset: {"interaction_vs_ordinary": by_seed} for dataset in transfer.SUPPORT_DATASETS}
        self.assertEqual(transfer.research_verdict(groups), "eligible_for_fresh_confirmation")
        by_seed[transfer.parallel.SEEDS[-1]]["correct"]["gain"] = -1 / 192
        self.assertEqual(transfer.research_verdict(groups), "no_go")
