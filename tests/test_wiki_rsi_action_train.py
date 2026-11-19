import unittest

import torch

from work3.p0_retrieval_diag import SUPPORT_DATASETS
from work3.wiki_rsi_action_model import EPOCHS, objectives, scheduled_indices
from work3.wiki_rsi_action_train import MIN_NET_WINS, summarize


class WikiActionTrainTest(unittest.TestCase):
    def test_objectives_share_positive_and_only_paired_penalizes_foil(self):
        scores = torch.tensor([-2.0, -1.0], requires_grad=True)
        positive, paired = objectives(scores, -1.0)
        self.assertAlmostEqual(positive.item(), 2.0)
        self.assertEqual(torch.autograd.grad(positive, scores, retain_graph=True)[0][1].item(), 0.0)
        self.assertGreater(torch.autograd.grad(paired, scores)[0][1].item(), 0.0)

    def test_fixed_schedule_uses_every_pair_in_each_epoch(self):
        schedule = scheduled_indices(155)
        self.assertEqual(schedule, scheduled_indices(155))
        self.assertEqual(len(schedule), 78)
        order = [index for group in schedule for index in group]
        self.assertEqual(len(order), EPOCHS * 155)
        self.assertTrue(all(order.count(index) == EPOCHS for index in range(155)))

    def test_decision_uses_both_literal_strata_and_both_controls(self):
        def case(ds, layer, positive, paired):
            return {"dataset": ds, "title_stratum": layer,
                    "arms": {arm: {"correct": value, "status": "executed"}
                             for arm, value in (("frozen", 0), ("positive_only", positive), ("paired", paired))}}

        rows = [case(ds, "literal", 0, 1) for ds in SUPPORT_DATASETS for _ in range(MIN_NET_WINS)]
        rows.append(case(SUPPORT_DATASETS[0], "not_literal", 1, 0))
        self.assertEqual(summarize(rows)[1], "confirm_on_new_natural_pages")
        rows[0] = case(SUPPORT_DATASETS[0], "literal", 1, 1)
        self.assertEqual(summarize(rows)[1], "no_go_for_current_paired_training")
