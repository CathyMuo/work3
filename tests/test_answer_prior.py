import importlib.util
import json
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from work3.answer_prior_text import (alias_regex, alias_token_strings, choose_replacement, clean_answer, contains,
                                     control_span, correct, has_answer, load_popqa, mentions, norm_answer, pair_facts,
                                     related, replacement_pool, split_contents, swap_item, swapped, title_key,
                                     token_string)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def record(qid, prop, obj, aliases=(), subject="Ann Lee", question=None, o_pop=10.0, s_pop=100.0):
    return {"qid": qid, "prop": prop, "obj": obj, "aliases": [obj, *aliases], "subj": subject,
            "question": question or f"What is {subject}'s {prop}?", "s_wiki_title": subject, "s_pop": s_pop,
            "o_pop": o_pop}


class TextRules(unittest.TestCase):
    def test_contents_split_and_title_key(self):
        self.assertEqual(split_contents('"Ann Lee"\nAnn Lee was a painter.'), ("Ann Lee", "Ann Lee was a painter."))
        self.assertEqual(split_contents("no title line"), (None, "no title line"))
        self.assertEqual(title_key("Henry_Feilden_(Conservative%20politician)"), "henry feilden (conservative politician)")

    def test_answer_rules(self):
        aliases = [norm_answer(value) for value in ("politician", "political leader")]
        self.assertTrue(correct("He was a politician.", aliases))
        self.assertFalse(correct("politicians", aliases))
        self.assertEqual(clean_answer("\nAnswer: The Beatles.\nmore"), "The Beatles")
        self.assertEqual(norm_answer("The Müller's Band!"), "muller band")

    def test_has_answer_is_token_based_and_accent_insensitive(self):
        doc = token_string('"X"\nBorn in São Paulo, Brazil.')
        self.assertTrue(has_answer(doc, alias_token_strings(["Sao Paulo"])))
        self.assertFalse(has_answer(doc, alias_token_strings(["Paul"])))

    def test_popqa_tsv(self):
        header = "id\tsubj\tprop\tobj\ts_wiki_title\ts_pop\to_pop\tquestion\tpossible_answers\n"
        line = ('1\tGeorge Rankin\toccupation\tpolitician\tGeorge Rankin\t142\t25692\tWhat is George Rankin\'s '
                'occupation?\t"[""politician"", ""political leader"", ""pol""]"\n')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.tsv"
            path.write_text(header + line, encoding="utf-8")
            rows, files, dropped = load_popqa(folder)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["aliases"], ["politician", "political leader", "pol"])
        self.assertEqual(rows[0]["s_pop"], 142.0)
        self.assertEqual(list(files), ["test.tsv"])


class Swaps(unittest.TestCase):
    def test_treatment_and_control_insert_the_same_entity(self):
        row = record("1", "occupation", "painter", subject="Ann Lee")
        contents = '"Ann Lee"\nAnn Lee (1901–1980) was an English painter who lived in Sussex. She was a painter.'
        item, reason = swap_item(row, contents, "sculptor")
        self.assertIsNone(reason)
        self.assertEqual(item["answer_mentions"], 2)
        self.assertNotIn("painter", split_contents(item["treat"])[1])
        self.assertEqual(split_contents(item["treat"])[0], "Ann Lee")
        self.assertIn("painter", item["control"])
        self.assertIn("sculptor", item["control"])
        self.assertNotIn(item["control_phrase"].lower(), {"ann", "lee", "she", "was"})

    def test_capitalised_answer_gets_capitalised_control(self):
        row = record("2", "place of birth", "Leeds", subject="Bob Hart", question="In what city was Bob Hart born?")
        contents = '"Bob Hart"\nBob Hart was born in Leeds and played for Arsenal in London.'
        item, reason = swap_item(row, contents, "Hull")
        self.assertIsNone(reason)
        self.assertIn(item["control_phrase"], {"Arsenal", "London"})
        self.assertIn("Leeds", item["control"])

    def test_leak_and_missing_span_are_reported(self):
        row = record("3", "country", "Mexico", subject="Mexico City", question="In what country is Mexico City?")
        self.assertEqual(swap_item(row, '"Mexico City"\nIt is in Mexico.', "Peru")[1], "alias_in_question_or_title")
        row = record("4", "occupation", "painter")
        self.assertEqual(swap_item(row, '"Ann Lee"\nAnn Lee was a poet.', "sculptor")[1], "no_alias_span")

    def test_control_span_excludes_answer_and_banned_tokens(self):
        text = "Ann Lee met Carl Orr in Paris."
        spans = mentions(text, alias_regex(["Paris"]))
        phrase, occurrences = control_span(text, spans, {"ann", "lee"})
        self.assertEqual(phrase, "Carl Orr")
        self.assertEqual(len(occurrences), 1)
        self.assertIsNone(control_span("Ann Lee in Paris.", mentions("Ann Lee in Paris.", alias_regex(["Paris"])),
                                       {"ann", "lee"}))

    def test_replacement_is_unrelated_and_absent(self):
        rows = [record(str(i), "genre", name, o_pop=float(i)) for i, name in
                enumerate(["rock music", "jazz", "pop music", "folk music", "blues"])]
        pools = replacement_pool(rows)
        self.assertEqual(sorted(pools["genre"]), sorted(row["obj"] for row in rows))
        target = record("9", "genre", "rock music", aliases=("rock",))
        choice = choose_replacement(target, pools["genre"], '"Band"\nThe Band played rock music and jazz.')
        self.assertNotIn(choice, {"rock music", "jazz"})
        self.assertFalse(related(choice, "rock"))

    def test_obscure_half_for_large_relations(self):
        rows = [record(str(i), "director", f"Person {i}", o_pop=float(i)) for i in range(40)]
        pool = replacement_pool(rows)["director"]
        self.assertTrue(all(int(name.split()[1]) <= 20 for name in pool))


class Pairing(unittest.TestCase):
    def facts(self):
        names = ["painter", "poet", "actor", "singer", "dancer", "lawyer"]
        result = []
        for index, name in enumerate(names):
            subject = f"Person{index} Smith"
            row = record(str(index), "occupation", name, subject=subject)
            row.update(stratum="known" if index < 4 else "unknown",
                       contents=f'"{subject}"\n{subject} was a well known {name} from Kent.')
            result.append(row)
        return result

    def test_pairs_stay_within_relation_and_stratum(self):
        pairs = pair_facts(self.facts())
        self.assertEqual(len(pairs), 3)
        for a, b in pairs:
            self.assertEqual((a["prop"], a["stratum"]), (b["prop"], b["stratum"]))
            self.assertIsNotNone(swapped(a["contents"], a, b["obj"]))

    def test_related_objects_never_pair(self):
        facts = self.facts()[:2]
        facts[1]["obj"], facts[1]["aliases"] = "painter", ["painter"]
        self.assertEqual(pair_facts(facts), [])


class Statistics(unittest.TestCase):
    def test_strata(self):
        from work3.answer_prior_stats import stratum

        aliases = [norm_answer("Paris")]
        self.assertEqual(stratum("Paris", ["Paris"] * 7 + ["Lyon"], aliases)["stratum"], "known")
        wrong = stratum("Lyon", ["Lyon"] * 6 + ["Nice", "Paris"], aliases)
        self.assertEqual(wrong["stratum"], "partial")  # one correct sample disqualifies 'wrong'
        self.assertEqual(stratum("Lyon", ["Lyon"] * 6 + ["Nice", "Metz"], aliases)["stratum"], "wrong")
        self.assertEqual(stratum("Lyon", ["Lyon", "Nice", "Metz", "Brest", "Lille", "Nantes", "Tours", "Pau"],
                                 aliases)["stratum"], "unknown")
        self.assertEqual(stratum("I don't know", ["a", "a", "a", "a", "b", "c", "d", "e"], aliases)["stratum"],
                         "unknown")

    def test_t1_gate(self):
        from work3.answer_prior_stats import MIN_N, t1

        rows = [{"qid": str(i), "stratum": "known", "rank_original": 1, "rank_treat": 9 if i % 2 else 2,
                 "rank_control": 2, "drop_treat": 0.02, "drop_control": 0.0} for i in range(MIN_N)]
        result = t1(rows)
        self.assertAlmostEqual(result["gap"], 0.5)
        self.assertTrue(result["pass"])
        rows.append({**rows[0], "rank_original": 7})
        self.assertEqual(t1(rows)["lost_top5_on_rescore"], 1)

    def test_matched_gap_reweights_to_wrong_cells(self):
        from work3.answer_prior_stats import matched_gap

        rows = []
        # cell A: unknown recall 1.0, wrong 0.5 (2 wrong); cell B: unknown 0.0, wrong 0.0 (2 wrong); cell C no unknown
        rows += [{"stratum": "unknown", "prop": "a", "pop_bin": 0, "hits": {10: True}}] * 3
        rows += [{"stratum": "wrong", "prop": "a", "pop_bin": 0, "hits": {10: flag}} for flag in (True, False)]
        rows += [{"stratum": "unknown", "prop": "b", "pop_bin": 0, "hits": {10: False}}]
        rows += [{"stratum": "wrong", "prop": "b", "pop_bin": 0, "hits": {10: False}}] * 2
        rows += [{"stratum": "wrong", "prop": "c", "pop_bin": 0, "hits": {10: False}}]
        self.assertAlmostEqual(matched_gap(rows, 10), 0.25)

    def test_t3_and_decision(self):
        from work3.answer_prior_stats import MIN_N, decide, t3

        directions = [{"pair": i // 2, "stratum": "known", "pref": 1.0 if i % 10 < 8 else 0.0} for i in range(MIN_N)]
        directions += [{"pair": 1000 + i // 2, "stratum": "unknown", "pref": 1.0 if i % 2 else 0.0}
                       for i in range(MIN_N)]
        result = t3(directions)
        self.assertAlmostEqual(result["pref_original"]["known"], 0.8)
        self.assertTrue(result["pass"])
        passing = {"T1": {"pass": True}, "T2": {"pass": False}, "T3": {"pass": True}, "valid": True}
        failing = {"T1": {"pass": True}, "T2": {"pass": False}, "T3": {"pass": False}, "valid": True}
        self.assertEqual(decide({"e5": failing, "qwen3": passing}, {"accuracy": 0.2})["verdict"], "GO")
        self.assertEqual(decide({"e5": failing, "qwen3": failing}, {"accuracy": 0.2})["verdict"], "NO-GO")
        self.assertEqual(decide({"e5": {**passing, "valid": False}}, {"accuracy": 0.2})["verdict"], "INCOMPLETE")
        self.assertEqual(decide({"e5": passing}, {"accuracy": 0.9})["verdict"], "INCOMPLETE")


class Pipeline(unittest.TestCase):
    """Probe construction and analysis on a synthetic corpus, with fake scores standing in for the encoders."""

    def test_probes_to_summary(self):
        import work3.answer_prior as ap

        objects = ["painter", "poet", "actor", "singer", "dancer", "lawyer", "farmer", "banker"]
        questions, contents, facts, strata = [], {}, {}, {}
        for index, obj in enumerate(objects):
            subject = f"Name{index} Smith"
            row = record(f"q{index}", "occupation", obj, subject=subject, s_pop=float(index))
            questions.append(row)
            contents[index] = f'"{subject}"\n{subject} was a {obj} who lived near the river in Kent.'
            contents[100 + index] = f'"Other{index}"\nAn unrelated passage number {index} about rivers.'
            facts[row["qid"]] = {"doc": index, "contents": contents[index], "leak": False}
            label = ("known", "known", "unknown", "unknown", "wrong", "wrong", "partial", "known")[index]
            strata[row["qid"]] = {"stratum": label, "answer": "teacher" if label == "wrong" else obj,
                                  "greedy_correct": label == "known", "samples_correct": 8 if label == "known" else 0,
                                  "mode_size": 8, "abstains": False}
        strata["q5"]["answer"] = "nurse"
        ranking = [[index] + [100 + other for other in range(len(objects)) if other != index] for index in range(8)]
        search = {"e5": ranking, "qwen3": ranking}
        context = {"replacement_pools": replacement_pool(questions), "facts": facts}
        probe, meta = ap.build_probes(context, questions, strata, search, contents)
        for name in ap.RETRIEVER_NAMES:
            self.assertEqual(len(probe[name]["t1"]), 8)
            self.assertEqual(len(probe[name]["t3"]), 6)  # known 3 facts -> 1 pair, unknown 1 pair, wrong 1 pair
            self.assertEqual(len(probe[name]["attraction"]), 2)
        self.assertEqual(meta["pairs"], 3)
        scores = {}
        for name in ap.RETRIEVER_NAMES:
            store = probe[name]
            values = [0.5] * len(store["pairs"])
            for item in store["t1"]:
                for position, index in enumerate(item["pool"]):
                    values[index] = 0.8 - 0.01 * position
                values[item["original"]], values[item["treat"]], values[item["control"]] = 0.9, 0.7, 0.89
            for item in store["t3"]:
                values[item["original"]], values[item["edited"]] = 0.9, 0.8
            scores[name] = values
        bm25 = [[100 + other for other in range(len(objects))] for _ in range(8)]
        search.update({"bm25": bm25, "e5+bm25": ranking, "qwen3+bm25": ranking})
        tokens = {doc: token_string(text) for doc, text in contents.items()}
        context.update(all_questions=questions)
        status = {name: {"parity": None} for name in ap.RETRIEVER_NAMES}
        meta_retrieval = {name: {"norm_min": 1.0, "norm_max": 1.0} for name in ap.RETRIEVER_NAMES}
        result, records = ap.analyse(context, questions, strata, search, probe, meta, scores, status,
                                     meta_retrieval, tokens)
        e5 = result["retrievers"]["e5"]
        self.assertEqual(e5["T1"]["n"], 8)
        self.assertEqual(e5["T1"]["dropout_treat"], 1.0)
        self.assertEqual(e5["T1"]["dropout_control"], 0.0)
        self.assertEqual(e5["T3"]["pref_original"]["known"], 1.0)
        self.assertEqual(e5["T2"]["recall"]["10"]["all"], 1.0)
        self.assertEqual(result["retrievers"]["bm25"]["T2"]["recall"]["10"]["all"], 0.0)
        self.assertEqual(result["verdict"], "NO-GO")  # every n is far below the frozen minimum
        self.assertEqual(Counter(record["kind"] for record in records), Counter({"t1": 16, "question": 8}))
        json.dumps(result)


class StageTimeline(unittest.TestCase):
    def test_stages_persist_and_reach_the_board(self):
        import time

        from work3.answer_prior import STAGE_PLAN, Stages
        from work3.wiki_rsi_progress import Progress, board_runs

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "answer-prior-smoke-v1"
            run.mkdir()
            progress = Progress(run, "answer_prior")
            stages = Stages(progress, STAGE_PLAN["run"])
            stages.start("preflight")
            stages.finish("preflight", note="160 题")
            stages.start("knownness", total=100)
            stages.advance("knownness", 40)
            stages.push(detail="answer_prior_knownness")
            progress.update("running", 1, 8, detail="answer_prior_knownness")  # no stages passed: kept
            row = board_runs(Path(folder), int(time.time()))[0]["phases"]["answer_prior"]
            keys = [stage["key"] for stage in row["stages"]]
            self.assertEqual(keys, [key for key, *_ in STAGE_PLAN["run"]])
            known = row["stages"][1]
            self.assertEqual((known["state"], known["done"], known["total"]), ("running", 40, 100))
            self.assertEqual(row["stages"][0]["note"], "160 题")
            self.assertTrue(row["stages"][4]["parallel"] and row["stages"][5]["join"])
            stages.fail_running()
            progress.update("failed", detail="answer_prior_incomplete", stages=stages.snapshot())
            row = board_runs(Path(folder), int(time.time()))[0]["phases"]["answer_prior"]
            self.assertEqual(row["stages"][1]["state"], "failed")

    def test_board_rejects_malformed_stages(self):
        from work3.wiki_rsi_progress import board_stages

        rows = [{"key": "ok_stage", "state": "running", "done": 5, "total": 10, "expected": 60},
                {"key": "Bad Key", "state": "running"}, {"key": "x", "state": "weird"},
                {"key": "y", "state": "done", "done": 11, "total": 10}, "nonsense"]
        self.assertEqual([row["key"] for row in board_stages(rows)], ["ok_stage"])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class CorpusAndBM25(unittest.TestCase):
    def test_scan_offsets_subjects_and_bm25(self):
        import numpy as np

        from work3.answer_prior_corpus import BM25, Corpus, bm25_terms, rrf, scan_corpus

        rows = [{"id": str(i), "contents": f'"Title {i % 7}"\nPassage {i} mentions river{i % 3} and town{i % 5}.'}
                for i in range(500)]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "wiki.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            vocabulary = sorted(set(bm25_terms("river1 town2")))
            summary = scan_corpus(path, vocabulary, {title_key("Title 3")}, folder / "prep", processes=2)
            self.assertEqual(summary["rows"], 500)
            self.assertEqual(summary["id_mismatch"], 0)
            self.assertEqual(summary["subject_docs"], sum(1 for i in range(500) if i % 7 == 3))
            corpus = Corpus(path, folder / "prep" / "offsets.npy")
            self.assertEqual(corpus.get(123), rows[123]["contents"])
            corpus.close()
            index = BM25(folder / "prep", vocabulary)
            ids, scores = index.search("river1 town2", 10)
            self.assertTrue(all(int(doc) % 3 == 1 and int(doc) % 5 == 2 for doc in ids[:5]))
            self.assertTrue(np.all(np.diff(scores) <= 0))
            self.assertEqual(rrf([[1, 2, 3], [3, 2, 9]], 3)[:2], [2, 3])

    def test_flat_index_header(self):
        import numpy as np

        from work3.answer_prior_corpus import flat_index_vectors

        vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
        header = b"IxFI" + struct.pack("<iqqq?i", 4, 3, 1 << 20, 1 << 20, True, 0) + struct.pack("<Q", 12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "e5_Flat.index"
            path.write_bytes(header + vectors.tobytes())
            loaded = flat_index_vectors(path)
            self.assertTrue(np.array_equal(np.asarray(loaded), vectors))
            path.write_bytes(header + vectors.tobytes()[:-4])
            self.assertIsNone(flat_index_vectors(path))


if __name__ == "__main__":
    unittest.main()
