import copy
import gzip
import json
import unittest

from work3.evidence_ref import question_key
from work3.wiki_rsi_references import V1, WIKI_RSI_REFERENCES, has_gap, repair_hotpot


class WikiReferenceTest(unittest.TestCase):
    def test_only_missing_support_is_filled(self):
        full = {"_id": "row1", "question": "Where?", "answer": "City", "type": "bridge",
                "supporting_facts": [["Page 1", 0], ["Page 2", 0]],
                "context": [["Page 1", ["The first."]]]}
        distractor = copy.deepcopy(full)
        distractor["context"].append(["Page 2", ["The second.", "A distractor sentence."]])
        key = question_key(full["question"])
        items = {key: {"type": "bridge", "titles": {"page 1": ["the first"], "page 2": [""]},
                       "answers": ["original answer hash"]}}
        repaired, counts = repair_hotpot(items, [full], [distractor])
        self.assertEqual(repaired[key]["titles"]["page 2"], ["the second"])
        self.assertEqual(repaired[key]["answers"], items[key]["answers"])
        self.assertEqual(counts, {"old_incomplete": 1, "new_incomplete": 0, "rows_changed": 1})
        distractor["supporting_facts"][1][1] = 1
        with self.assertRaisesRegex(ValueError, "Official HotpotQA fields"):
            repair_hotpot(items, [full], [distractor])

    def test_pack_preserves_benchmark_identity(self):
        with gzip.open(V1, "rt", encoding="utf-8") as stream:
            old = json.load(stream)["datasets"]
        with gzip.open(WIKI_RSI_REFERENCES, "rt", encoding="utf-8") as stream:
            new = json.load(stream)["datasets"]
        self.assertEqual(old["2wikimultihopqa"], new["2wikimultihopqa"])
        self.assertEqual(old["hotpotqa"]["items"].keys(), new["hotpotqa"]["items"].keys())
        self.assertEqual(sum(has_gap(row["titles"]) for row in new["hotpotqa"]["items"].values()), 2)
        for key, row in old["hotpotqa"]["items"].items():
            replacement = new["hotpotqa"]["items"][key]
            self.assertEqual(row["answers"], replacement["answers"])
            self.assertEqual(row["titles"].keys(), replacement["titles"].keys())
            if not has_gap(row["titles"]):
                self.assertEqual(row, replacement)
