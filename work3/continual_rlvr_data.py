"""Data preparation and deterministic rewards for the offline continual-RLVR pilot."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from work3.wiki_rsi_progress import atomic_json

SOURCES = {"a": "allenai/RLVR-MATH", "b": "allenai/RLVR-IFeval"}
REVISIONS = {"a": "bd2a93551b503a395fadd1a740d957559cfe6f3c", "b": "47c03c73621c4aab2b824b7818681117d662770e"}
PARQUET_SHA = {"a": "2fb8483bf7a4ae9821bee1e715490fb8de5b59f0d2b58bd4de78cc75094d9422",
               "b": "75f1b1f63039044034fe75d22f51f77f257a79291d0084262bb43695d4aac971"}
PARQUET_NAME = "train-00000-of-00001.parquet"
SIZES = {"a_train": 2048, "a_dev": 128, "b_anchor": 128, "b_train": 768, "b_dev": 256}
PROMPT_BOUND = 768
BUNDLE_VERSION = 2
LITERAL_RULES = {"validate_quotation", "validate_title", "validate_json_format"}
WORDS = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def sha256_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(text):
    return " ".join(text.casefold().split())


def load_prompt_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(model).resolve(strict=True), local_files_only=True, use_fast=True)


def tokenizer_fingerprint(model):
    model = Path(model)
    filenames = ("tokenizer.json", "tokenizer_config.json")
    optional = ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
    paths = [model / name for name in filenames]
    paths.extend(model / name for name in optional if (model / name).is_file())
    return {path.name: sha256_file(path) for path in paths}


def prompt_ids(tokenizer, case):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}], tokenize=True, add_generation_prompt=True)


def prompt_audit(splits, tokenizer):
    maxima = {}
    for name, cases in splits.items():
        max_tokens = 0
        for case in cases:
            count = len(prompt_ids(tokenizer, case))
            if not 1 <= count <= PROMPT_BOUND:
                raise ValueError(f"{name} prompt {case['id']} has {count} tokens, expected 1–{PROMPT_BOUND}")
            max_tokens = max(max_tokens, count)
        maxima[name] = max_tokens
    return maxima


def rational(value):
    value = value.strip().strip("$").strip().replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\,", "").replace(" ", "")
    if len(value) > 120:
        return None
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
        value = value.replace(",", "")
    latex = re.fullmatch(r"([+-]?)\\(?:d|t)?frac\{([+-]?\d+)\}\{([+-]?\d+)\}", value)
    if latex:
        sign, numerator, denominator = latex.groups()
        return (-1 if sign == "-" else 1) * Fraction(int(numerator), int(denominator)) if int(denominator) else None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+|\d+/\d+)", value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def last_box(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return ""
    start += len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if not depth:
            return text[start:i]
    return ""


def rule_valid(rule):
    name = rule.get("func_name")
    n = rule.get("N")
    word = lambda key: isinstance(rule.get(key), str) and bool(rule[key].strip())
    integer = type(n) is int and n > 0
    if name == "verify_keywords":
        terms = rule.get("keyword_list")
        return isinstance(terms, list) and bool(terms) and all(isinstance(t, str) and t.strip() for t in terms)
    if name == "verify_keyword_frequency":
        return integer and n <= 12 and word("word")
    if name == "validate_word_constraint":
        return integer and 20 <= n <= 125 and rule.get("quantifier") in {"at least", "at most"}
    if name in {"verify_bullet_points", "validate_placeholders"}:
        return integer and n <= 10
    return name in LITERAL_RULES or name == "validate_end" and word("end_phrase")


def render_constraint(template, rule):
    keys = {"{N}": "N", "{word}": "word", "{end phrase}": "end_phrase"}
    for placeholder, key in keys.items():
        if placeholder in template and rule.get(key) is not None:
            template = template.replace(placeholder, str(rule[key]))
    if rule["func_name"] == "verify_keywords":
        terms = rule["keyword_list"]
        for index, term in enumerate(terms, start=1):
            template = template.replace("{keyword" + str(index) + "}", term)
    if rule["func_name"] == "validate_word_constraint":
        template = template.replace("at least / around / at most", rule["quantifier"])
    return template


def score(case, text):
    if case["task"] == "a":
        answer = rational(last_box(text))
        return answer is not None and str(answer) == case["target"]
    if not text.strip():
        return False
    rule = case["target"]
    name, n = rule["func_name"], rule.get("N")
    if name == "verify_keywords":
        return all(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.I)
                   for term in rule["keyword_list"])
    if name == "verify_keyword_frequency":
        return len(re.findall(r"(?<!\w)" + re.escape(rule["word"]) + r"(?!\w)", text, flags=re.I)) == n
    if name == "validate_word_constraint":
        count = len(WORDS.findall(text))
        return count >= n if rule["quantifier"] == "at least" else count <= n
    if name == "verify_bullet_points":
        return len(re.findall(r"(?m)^[ \t]*[-*+][ \t]+\S", text)) == n
    if name == "validate_placeholders":
        return len(re.findall(r"\[[^\[\]\n]+\]", text)) >= n
    if name == "validate_end":
        ending = rule["end_phrase"].strip()
        boundary = r"(?<!\w)" if ending[0].isalnum() else ""
        return re.search(boundary + re.escape(ending) + r"\Z", text.rstrip(), flags=re.I) is not None
    if name == "validate_quotation":
        return len(text.strip()) > 2 and text.strip().startswith('"') and text.strip().endswith('"')
    if name == "validate_title":
        return re.search(r"<<[^<>\n]+>>", text) is not None
    if name == "validate_json_format":
        try:
            return isinstance(json.loads(text), (dict, list))
        except (ValueError, RecursionError):
            return False
    raise ValueError(f"unsupported verifier: {name}")


def normalize(task, raw):
    messages = raw["messages"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("unexpected upstream message schema")
    original = messages[0]["content"].strip()
    if task == "a":
        answer = rational(raw["ground_truth"])
        if answer is None:
            return None
        question = re.split(r"(?:^|\n\n)Question:\s*", original)[-1].strip()
        if not question:
            return None
        prompt = question + "\n\nSolve the problem and put the exact rational final answer in \\boxed{}."
        target = str(answer)
        group = digest(canonical(question))
    else:
        target = json.loads(raw["ground_truth"])
        if not isinstance(target, dict) or not rule_valid(target) or not original:
            return None
        prompt = original
        constraint = render_constraint((raw.get("constraint") or "").strip(), target)
        normalized = canonical(original)
        needle = canonical(constraint)
        if not needle or normalized.count(needle) != 1:
            return None
        group = digest(canonical(normalized.replace(needle, " ", 1)))
    if len(prompt) > 1800:
        return None
    return {"id": digest(canonical(prompt)), "group": group, "task": task, "prompt": prompt, "target": target}


def split_rows(rows, task, tokenizer):
    groups = defaultdict(list)
    for raw in rows:
        case = normalize(task, raw)
        if case:
            groups[case["group"]].append(case)
    # Keep one stable variant per source question, including across rules.
    pool = []
    for members in groups.values():
        case = min(members, key=lambda row: (row["id"], json.dumps(row["target"], sort_keys=True)))
        if any(row["id"] == case["id"] and row["target"] != case["target"] for row in members):
            continue
        pool.append(case)
    pool.sort(key=lambda row: digest("work3-c-v1:" + row["id"]))
    sizes = {name: count for name, count in SIZES.items() if name.startswith(task)}
    required = sum(sizes.values())
    if len(pool) < required:
        raise ValueError(f"task {task} has {len(pool)} eligible source questions, needs {required}")
    offset = 0
    result, maxima, replacements = {}, {}, []
    reserve = iter(pool[required:])
    # Keep the original positions of valid cases across all splits. Replacements
    # come only from groups outside those splits, in the same frozen order.
    for name in sorted(sizes, key=lambda name: not name.endswith("_dev")):
        selected = []
        for case in pool[offset:offset + sizes[name]]:
            count = len(prompt_ids(tokenizer, case))
            if not 1 <= count <= PROMPT_BOUND:
                if not name.endswith("_train"):
                    raise ValueError(f"{name} prompt {case['id']} has {count} tokens; frozen holdout cannot change")
                removed_id, removed_tokens = case["id"], count
                for candidate in reserve:
                    count = len(prompt_ids(tokenizer, candidate))
                    if 1 <= count <= PROMPT_BOUND:
                        case = candidate
                        break
                else:
                    raise ValueError(f"task {task} has insufficient token-eligible reserve questions")
                replacements.append({
                    "split": name, "removed_id": removed_id, "removed_tokens": removed_tokens,
                    "replacement_id": case["id"], "replacement_tokens": count,
                })
            selected.append(case)
            maxima[name] = max(maxima.get(name, 0), count)
        result[name] = selected
        offset += sizes[name]
    return result, maxima, replacements


def load_bundle(path):
    with Path(path).open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    if bundle["protocol"] != "continual_rlvr_v1" or set(bundle["splits"]) != set(SIZES):
        raise ValueError("unsupported data bundle")
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("expected data bundle version 2; repack both original parquet files using --model")
    policy = bundle["prompt_policy"]
    if (policy["max_tokens"] != PROMPT_BOUND or set(policy["split_max_tokens"]) != set(SIZES)
            or any(type(count) is not int or not 1 <= count <= PROMPT_BOUND
                   for count in policy["split_max_tokens"].values())):
        raise ValueError("invalid prompt budget in data bundle")
    if set(bundle["sources"]) != set(SOURCES):
        raise ValueError("missing data provenance")
    for task, source in bundle["sources"].items():
        if source["name"] != SOURCES[task] or source["revision"] != REVISIONS[task]:
            raise ValueError(f"unlocked source {task}")
        if source["files"] != {PARQUET_NAME: PARQUET_SHA[task]}:
            raise ValueError(f"wrong frozen data checksum for {task}")
    ids, groups = set(), set()
    for name, cases in bundle["splits"].items():
        if len(cases) != SIZES[name]:
            raise ValueError(f"{name} size drift")
        for case in cases:
            identifier, group = case["id"], case["group"]
            task = case["task"]
            if (task != name[0] or not case["prompt"].strip() or identifier != digest(canonical(case["prompt"]))
                    or not re.fullmatch(r"[0-9a-f]{64}", group) or identifier in ids or group in groups):
                raise ValueError(f"invalid or overlapping case in {name}")
            if ((task == "a" and rational(case["target"]) is None)
                    or (task == "b" and not rule_valid(case["target"]))):
                raise ValueError(f"invalid verifier in {name}")
            ids.add(identifier)
            groups.add(group)
    return bundle


def pack(args):
    from pyarrow.parquet import ParquetFile

    tokenizer = load_prompt_tokenizer(args.model)
    sources, splits, maxima, replacements = {}, {}, {}, []
    for task in SOURCES:
        revision = getattr(args, task + "_revision")
        path = getattr(args, task)
        if revision != REVISIONS[task] or sha256_file(path) != PARQUET_SHA[task]:
            raise ValueError(f"task {task} requires its frozen upstream revision and parquet")
        rows = (row for batch in ParquetFile(path).iter_batches() for row in batch.to_pylist())
        selected, task_maxima, task_replacements = split_rows(rows, task, tokenizer)
        splits.update(selected)
        maxima.update(task_maxima)
        replacements.extend(task_replacements)
        sources[task] = {"name": SOURCES[task], "revision": revision,
                         "files": {PARQUET_NAME: PARQUET_SHA[task]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy = {"max_tokens": PROMPT_BOUND, "tokenizer_sha256": tokenizer_fingerprint(args.model),
              "split_max_tokens": maxima, "replacements": replacements}
    atomic_json(args.out, {"protocol": "continual_rlvr_v1", "bundle_version": BUNDLE_VERSION,
                          "sources": sources, "prompt_policy": policy, "splits": splits})
    load_bundle(args.out)
    print(json.dumps({"sha256": sha256_file(args.out), "sizes": SIZES, "prompt_policy": policy}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for task in SOURCES:
        parser.add_argument(f"--{task}", type=Path, required=True, help="local train split parquet")
        parser.add_argument(f"--{task}-revision", required=True)
    parser.add_argument("--model", type=Path, required=True, help="local Qwen2.5-3B-Instruct snapshot")
    parser.add_argument("--out", required=True, type=Path)
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
