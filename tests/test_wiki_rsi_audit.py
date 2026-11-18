import json
import tempfile
import unittest
from pathlib import Path

from work3.evidence_ref import question_key, text_key
from work3.p0_retrieval_diag import SUPPORT_DATASETS
from work3.wiki_rsi_audit import AUDIT_V1, SEED, collect, validate_baseline
from work3.wiki_rsi_replay import catalog


def record(question, passages, score=0):
    call = {"name": "search", "arguments": {"query_list": ["some query"]}}
    body = "\n".join(f"Doc {i} (Title: {title})\n{text}" for i, (title, text) in enumerate(passages, 1))
    reply = {"result": body}
    return {"input": "Question: " + question, "data_source": "hotpotqa", "score": score,
            "output": f"<tool_call>{json.dumps(call)}</tool_call>"
                      f"<tool_response>{json.dumps(reply)}</tool_response>"}


class WikiRsiAuditTest(unittest.TestCase):
    def test_missing_requires_actual_sentence_and_source_coordinates(self):
        rows = [
            record("eligible", [("First", "The bridge sentence."), ("Second", "Other contents.")]),
            record("source", [("Second", "The missing sentence.")]),
            record("complete", [("First", "The bridge sentence."), ("Second", "The missing sentence.")]),
            record("unlocated", [("First", "The bridge sentence.")]),
            record("duplicate", [("First", "The bridge sentence.")]),
            record("duplicate", [("First", "The bridge sentence.")]),
            {**record("bad", []), "output": "<tool_response>"},
        ]
        tests = {name: {} for name in SUPPORT_DATASETS}
        refs = {name: {} for name in SUPPORT_DATASETS}
        for question in ("eligible", "complete", "unlocated", "duplicate", "bad"):
            key = question_key(question)
            tests["hotpotqa"][key] = ["Example"]
            second = "A sentence which never occurred." if question == "unlocated" else "The missing sentence."
            refs["hotpotqa"][key] = {"titles": {"first": [text_key("The bridge sentence.")],
                                               "second": [text_key(second)]}}
        with tempfile.TemporaryDirectory() as folder:
            dump = Path(folder) / "eval.jsonl"
            dump.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary, manifest = collect(dump, tests, refs)
        result = summary["hotpotqa"]
        self.assertEqual(result["dump_rows"], 7)
        self.assertEqual(result["trace_rejected:unbalanced_tags"], 1)
        self.assertEqual(result["missing_without_all_source_passages"], 1)
        self.assertEqual(result["duplicate_question_rows"], 2)
        self.assertEqual(result["missing_with_source"], 1)
        self.assertEqual(result["evidence_present"], 1)
        case = next(row for row in manifest if row["cohort"] == "missing_with_source")
        self.assertEqual(case["question_key"], question_key("eligible"))
        self.assertEqual(case["support_passages"], [(2, 1, 1)])

    def test_cache_expansion_preserves_target_and_uses_most_recent_real_source(self):
        key = question_key("target")
        tests = {ds: {} for ds in SUPPORT_DATASETS}
        refs = {ds: {} for ds in SUPPORT_DATASETS}
        tests["hotpotqa"][key] = ["City"]
        refs["hotpotqa"][key] = {"titles": {
            "title a": ["bridge sentence"], "title b": ["missing sentence", "missing third fact"],
        }}
        with tempfile.TemporaryDirectory() as folder:
            def dump(name, rows):
                path = Path(folder) / name
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                return path
            target = dump("iter3", [record("target", [("Title A", "Bridge sentence.")]),
                                    record("other", [("Title B", "Missing sentence.")], 1)])
            corrupt = record("broken", [("Title B", "Missing third fact.")], 1)
            corrupt["output"] += "</tool_response>"
            caches = {
                2: dump("iter2", [corrupt, record("other", [("Title B", "Missing third fact.")], 1)]),
                1: dump("iter1", [record("other", [("Title B", "Missing third fact.")], 1)]),
                0: dump("iter0", [record("other", [("Title B", "Missing sentence.")], 1)]),
            }
            before, _ = collect(target, tests, refs)
            after, manifest = collect(target, tests, refs, caches)
            provenance = {"dump_sha256": "target", "test_sha256": "test", "references_sha256": "v2"}
            baseline = {"protocol": AUDIT_V1, "seed": SEED, "provenance": provenance, "by_dataset": before}
            expanded = {"provenance": provenance, "by_dataset": after}
            validate_baseline(expanded, baseline)
            case = manifest[0]
            self.assertEqual(case["dump_line"], 1)
            self.assertEqual(case["earliest_source_iteration"], 2)
            self.assertEqual(case["support_passages"], [(2, 2, 1, 1), (3, 2, 1, 1)])
            rows, passages, negatives = catalog(target, manifest, caches)
            self.assertEqual(rows[1]["input"], "Question: target")
            self.assertEqual(passages[("hotpotqa", 2, 2, 1, 1)], ("Title B", "Missing third fact."))
            self.assertNotIn(("title b", "missing third fact"), negatives["hotpotqa"])
            after["hotpotqa"]["originally_wrong"] = 2
            with self.assertRaisesRegex(ValueError, "Same-round audit counts changed"):
                validate_baseline(expanded, baseline)
