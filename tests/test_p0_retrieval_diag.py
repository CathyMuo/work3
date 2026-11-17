"""Regression checks for query-budget accounting and real annotation schemas."""

import json
import unittest

from work3 import p0_retrieval_diag as diag
from work3.evidence_ref import prepare_reference


def tag(kind, payload):
    return f"<{kind}>{json.dumps(payload)}</{kind}>"


def call(*queries):
    return tag("tool_call", {"name": "search", "arguments": {"query_list": queries}})


def response(*blocks):
    return tag("tool_response", {"result": "\n---\n".join(blocks)})


class RetrievalDiagnosticTest(unittest.TestCase):
    def ref(self, dataset):
        field = "sentences" if dataset == "hotpotqa" else "content"
        qtype = "bridge" if dataset == "hotpotqa" else "compositional"
        return prepare_reference(dataset, {
            "golden_answers": ["Paris"],
            "metadata": {"type": qtype, "supporting_facts": {"title": ["Film", "Director"], "sent_id": [0, 0]},
                         "context": {"title": ["Film", "Director"], field: [
                             ["The film was directed by Mira."], ["Mira was born in Paris."]]}},
        })

    def test_multiquery_first_call_keeps_distinct_query_budgets(self):
        first = 'Doc 1 (Title: "Film")\nThe film was directed by Mira.'
        second = 'Doc 1 (Title: Director)\nMira was born in Paris.'
        output = call("film", "director") + response(first, second)
        for dataset in diag.SUPPORT_DATASETS:
            with self.subTest(dataset=dataset):
                ref = self.ref(dataset)
                self.assertEqual((ref["bridge_title"], ref["answer_title"]), ("film", "director"))
                row = diag.observe({"output": output, "score": 1}, ["Paris"], ref)
                self.assertEqual((row.calls, row.queries, row.first_call), (1, 2, 2))
                self.assertEqual(row.hit["all_titles"], 2)
                self.assertEqual(row.hit["all_sentences"], 2)
                report = diag.summarize([row])
                self.assertEqual(report["all_titles"]["by_queries"], {"1": 0, "2": 1, "3": 1, "5": 1})
                self.assertEqual(report["all_titles"]["first_call"], 1)

    def test_bad_block_count_fails_and_known_empty_counts(self):
        with self.assertRaisesRegex(diag.TraceError, "query_block_count"):
            diag.trace_of(call("one", "two") + response("Doc 1 (Title: One)\nevidence"))
        trace = diag.trace_of(call("one", "two") + response("No search results found."))
        self.assertEqual((trace.queries, trace.first_call, trace.errors), ([[], []], 2, 0))
        trace = diag.trace_of(call("one") + response("Search error: timeout"))
        self.assertEqual((trace.queries, trace.errors), ([[]], 1))

    def test_missing_result_and_natural_stop_are_distinct(self):
        with self.assertRaisesRegex(diag.TraceError, "call_without_response"):
            diag.trace_of(call("query"))
        row = diag.observe({"output": "<answer>Paris</answer>", "score": 1}, ["Paris"], self.ref("hotpotqa"))
        self.assertEqual(row.queries, 0)
        self.assertIsNone(row.hit["all_titles"])
        self.assertEqual(diag.summarize([row])["all_titles"]["by_queries"]["2"], 0)

    def test_title_and_exact_sentence_are_separate(self):
        output = call("first") + response("Doc 1 (Title: Film)\nThe film was released in 1994.")
        output += call("second") + response("Doc 1 (Title: Director)\nMira bought a house in Paris.")
        row = diag.observe({"output": output, "score": 0}, ["Paris"], self.ref("hotpotqa"))
        self.assertEqual(row.hit["all_titles"], 2)
        self.assertIsNone(row.hit["all_sentences"])
        self.assertEqual(diag.summarize([row])["support_titles"]["mean_new_after_first_call"], 1)

    def test_gate_needs_both_benchmarks_and_support_labels(self):
        counts = {ds: diag.FLASHRAG_COUNTS[ds] for ds in diag.SUPPORT_DATASETS}
        inputs = {i: {ds: {"dump_rows": n} for ds, n in counts.items()} for i in ("1", "3")}
        neg = diag.paired_stats((100, 90, 10, 80))
        zero = diag.paired_stats((0, 0, 0, 0))
        paired = {ds: {"paired": n, "all_titles_q2": diag.paired_stats((n, n, 0, n)),
                       "all_sentences_q2": zero, "bridge_joint_q1_miss_q3": neg}
                  for ds, n in counts.items()}
        verdict = diag.decide(inputs, counts, paired)
        self.assertEqual(verdict["verdict"], "GO_PREFIX_REPLAY")
        self.assertEqual(verdict["replay_priority"], "bridge_recovery")
        paired["2wikimultihopqa"]["all_titles_q2"] = zero
        self.assertEqual(diag.decide(inputs, counts, paired)["verdict"], "INVALID_INPUT")
