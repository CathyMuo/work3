"""Build disjoint feedback and scoring pools for ordinary Python algorithm tasks."""

import ast
import hashlib
import itertools
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

from work3.continual_rlvr_data import sha256_file


PROTOCOL = "feedback_pair_v1"
DATA_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"
TARGET = 384
DRAW_CAP = 1152  # per split, after CPU reference checks
SCORE_CASES = 8
FEEDBACK_CASES = 12
TEXT_CAP = 192
SHARD = re.compile(r"train-(\d{5})-of-00039-[a-f0-9]{16}\.parquet")
FIELDS = ("name", "description", "cf_contest_id", "solutions", "public_tests", "private_tests",
          "generated_tests", "input_file", "output_file")
MULTIPLE_OUTPUTS = re.compile(r"(?:print|output) any (?:one|valid|possible|of)|multiple (?:correct|valid) (?:answers|solutions)", re.I)


def key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text):
    return " ".join(text.split())


def similar_input(a, b):
    """Group exact inputs and single replacement neighbors with the same line shape."""
    a_lines, b_lines = [line.split() for line in a.strip().splitlines()], [line.split() for line in b.strip().splitlines()]
    if list(map(len, a_lines)) != list(map(len, b_lines)):
        return normalized(a) == normalized(b)
    return sum(x != y for x, y in zip(itertools.chain.from_iterable(a_lines),
                                       itertools.chain.from_iterable(b_lines))) <= 1


def test_cases(raw):
    return [(i, normalized(o)) for i, o in zip(raw["input"], raw["output"], strict=True)
            if 0 < len(i.strip()) <= TEXT_CAP and 0 < len(o.strip()) <= TEXT_CAP]


def split_tests(row):
    public = test_cases(row["public_tests"])
    feedback, scoring = [], []
    witnesses = {}
    for case in public + test_cases(row["generated_tests"]) + test_cases(row["private_tests"]):
        fingerprint = normalized(case[0])
        if fingerprint in witnesses and witnesses[fingerprint] != case[1]:
            return [], []
        witnesses[fingerprint] = case[1]
    for case in sorted(test_cases(row["generated_tests"]), key=lambda t: key(t[0])):
        if not any(similar_input(case[0], other[0]) for other in public + feedback):
            feedback.append(case)
        if len(feedback) == FEEDBACK_CASES:
            break
    for case in sorted(test_cases(row["private_tests"]), key=lambda t: key(t[0])):
        if not any(similar_input(case[0], other[0]) for other in public + feedback + scoring):
            scoring.append(case)
        if len(scoring) == SCORE_CASES:
            break
    return feedback, scoring


def clean_code(text):
    fenced = re.fullmatch(r"\s*```(?:python|py)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*", text, re.I)
    code = (fenced.group(1) if fenced else text).strip()
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    return code if tree.body else None


def read_rows(source):
    from pyarrow.parquet import ParquetFile

    shards = sorted((source / "data").glob("train-*.parquet"))
    matches = [SHARD.fullmatch(path.name) for path in shards]
    if any(match is None for match in matches) or [int(m[1]) for m in matches] != list(range(39)):
        raise ValueError("expected all 39 original train shards under SOURCE/data")
    kept = {}
    fingerprints = {}
    for path in shards:
        fingerprints[path.name] = sha256_file(path)
        for batch in ParquetFile(path).iter_batches(batch_size=16, columns=FIELDS):
            for raw in batch.to_pylist():
                task = task_from_raw(raw)
                if task is None:
                    continue
                # Group formatting and number-only variants by the normalized opening.
                opening = re.findall(r"[a-z]+|[0-9]+", raw["description"].lower())[:100]
                fingerprint = key(" ".join("#" if word.isdigit() else word for word in opening))
                previous = kept.get(fingerprint)
                if previous is None or task["id"] < previous["id"]:
                    kept[fingerprint] = task
    return sorted(kept.values(), key=lambda row: row["id"]), fingerprints


def task_from_raw(row):
    description = row["description"].strip()
    contest = row["cf_contest_id"]
    if (not contest or contest < 1 or row["input_file"] or row["output_file"] or not description
            or len(description) > 4200 or MULTIPLE_OUTPUTS.search(description)):
        return None
    sources = row["solutions"]
    refs = [code for language, code in zip(sources["language"], sources["solution"], strict=True)
            if language == 3 and len(code) <= 12000 and clean_code(code)]
    if not refs:
        return None
    feedback, scoring = split_tests(row)
    if len(feedback) < 4 or len(scoring) != SCORE_CASES or not test_cases(row["public_tests"]):
        return None
    split = "train" if int(key(f"{PROTOCOL}:{contest}"), 16) % 2 == 0 else "sealed"
    return {"id": key(f"{contest}:{row['name']}:{description}"), "contest": contest, "split": split,
            "description": description, "feedback": feedback, "score": scoring,
            "refs": sorted(set(refs), key=key)[:2]}


class Verifier:
    """Run each submission against one input at a time in an isolated Linux process."""

    LIMITS = (
        "import resource,runpy,sys; "
        "[resource.setrlimit(k,(v,v)) for k,v in [(resource.RLIMIT_CPU,1),"
        "(resource.RLIMIT_AS,268435456),(resource.RLIMIT_FSIZE,65536),(resource.RLIMIT_NOFILE,32),"
        "(resource.RLIMIT_NPROC,1),(resource.RLIMIT_CORE,0)]]; "
        "sys.stderr.write('WORK3_SANDBOX_READY\\n'); sys.stderr.flush(); "
        "runpy.run_path('/submission.py',run_name='__main__')"
    )

    def __init__(self):
        self.bwrap = shutil.which("bwrap")
        if os.name != "posix" or not self.bwrap or not Path("/usr/bin/python3").is_file():
            raise RuntimeError("Linux with Bubblewrap and /usr/bin/python3 is required for submission execution")
        self.base = [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                     "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "LC_ALL", "C.UTF-8",
                     "--ro-bind", "/usr", "/usr", "--ro-bind-try", "/lib", "/lib",
                     "--ro-bind-try", "/lib64", "/lib64", "--dev", "/dev", "--proc", "/proc"]

    def evaluate(self, code, cases):
        if code is None:
            return [("invalid", "") for _ in cases], 0, 0.0
        started = monotonic()
        with tempfile.TemporaryDirectory(prefix="work3-pair-") as folder:
            path = Path(folder) / "submission.py"
            path.write_text(code + "\n", encoding="utf-8")
            command = self.base + ["--ro-bind", str(path), "/submission.py", "--remount-ro", "/",
                                   "--chdir", "/", "--", "/usr/bin/python3", "-I", "-B", "-S", "-c", self.LIMITS]
            results = [self.run_case(command, given, expected) for given, expected in cases]
        return results, len(cases), monotonic() - started

    @staticmethod
    def run_case(command, given, expected):
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            timed_out = False
            try:
                result = subprocess.run(command, input=given.encode("utf-8"), stdout=stdout,
                                        stderr=stderr, check=False, timeout=2.5)
            except subprocess.TimeoutExpired:
                timed_out = True
            stderr.seek(0)
            marker = stderr.readline(256)
            if marker != b"WORK3_SANDBOX_READY\n":
                raise RuntimeError(f"isolated Python did not start: {marker.decode('utf-8', errors='replace').strip()}")
            if timed_out:
                return "timeout", ""
            stdout.seek(0)
            output = normalized(stdout.read(65537).decode("utf-8", errors="replace"))
        if result.returncode != 0:
            return "error", ""
        return ("pass" if output == expected else "wrong"), output[:TEXT_CAP + 1]

    def probe(self):
        namespace = os.readlink("/proc/self/ns/net")
        code = ("import os, sys\nassert not os.path.exists('/home')\n"
                f"assert os.readlink('/proc/self/ns/net') != {namespace!r}\nprint(sys.stdin.read().strip())")
        if self.evaluate(code, [("work3-isolation\n", "work3-isolation")])[0] != [("pass", "work3-isolation")]:
            raise RuntimeError("Bubblewrap failed the actual isolated execution probe")

    def version(self):
        return {"bubblewrap": subprocess.check_output([self.bwrap, "--version"], text=True, timeout=5).strip(),
                "python": subprocess.check_output(["/usr/bin/python3", "--version"], text=True, timeout=5).strip(),
                "bubblewrap_sha256": sha256_file(Path(self.bwrap)),
                "python_sha256": sha256_file(Path("/usr/bin/python3"))}


def check_references(tasks, verifier, jobs, report):
    selected = defaultdict(list)
    checked = 0

    def validate(task):
        cases = task["feedback"] + task["score"]
        return all(all(status == "pass" for status, _ in verifier.evaluate(clean_code(ref), cases)[0])
                   for ref in task["refs"])

    with ThreadPoolExecutor(max_workers=jobs) as thread_pool:
        for start in range(0, len(tasks), jobs):
            batch = [task for task in tasks[start:start + jobs] if len(selected[task["split"]]) < DRAW_CAP]
            for task, valid in zip(batch, thread_pool.map(validate, batch), strict=True):
                checked += 1
                if valid and len(selected[task["split"]]) < DRAW_CAP:
                    selected[task["split"]].append({k: v for k, v in task.items() if k != "refs"})
            report(start + min(jobs, len(tasks) - start), len(tasks), sum(map(len, selected.values())))
            if all(len(selected[split]) == DRAW_CAP for split in ("train", "sealed")):
                break
    return dict(selected), checked
