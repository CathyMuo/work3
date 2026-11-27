"""Text rules for the answer-prior kill test: normalisation, PopQA records, alias spans, swaps and pairing.

Frozen protocol: research/protocols/answer_prior_p0.md. Standard library only, so every rule is unit-tested on CPU.
"""

import ast
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

PROTOCOL = "answer_prior_p0_v1"
COMBINING = re.compile(r"[̀-ͯ]")
POSSESSIVE = re.compile(r"['’]s\b")
TOKEN = re.compile(r"\w+")
PUNCT = re.compile(r"[^\w\s]")
SPACE = re.compile(r"\s+")
# Lucene's default English stop set, used for BM25 only.
BM25_STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                      "there these they this to was will with".split())
# Words that never make a control span (sentence-initial capitals, pronouns, function words, months).
CONTROL_STOP = BM25_STOP | frozenset(
    "he she his her him its was were has had have who whom whose which what when where while after before during "
    "from also other since until about over under between among within without january february march april may "
    "june july august september october november december".split())
ABSTAIN = re.compile(r"\b(unknown|not sure|no idea|cannot|can't|dont know|don't know|do not know|no information|"
                     r"not known|unclear|i'm sorry|i am sorry)\b", re.IGNORECASE)
CAPITAL_SPAN = re.compile(r"[A-Z][\w'\-.]*(?:\s+[A-Z][\w'\-.]*)*")
LOWER_WORD = re.compile(r"(?<![\w-])[a-z][a-z\-]{3,}(?![\w-])")
REQUIRED = ("question", "prop", "obj", "s_wiki_title", "possible_answers")


def digest(*parts):
    return hashlib.sha256(":".join(str(part) for part in (PROTOCOL, *parts)).encode("utf-8")).hexdigest()


def fold(text):
    """Accent-insensitive, case-folded text (C-level operations only; runs over 21M passages)."""
    return COMBINING.sub("", unicodedata.normalize("NFKD", text)).casefold()


def tokens(text):
    return TOKEN.findall(POSSESSIVE.sub("", fold(text)))


def token_string(text):
    return " " + " ".join(tokens(text)) + " "


def has_answer(doc_tokens, alias_tokens):
    """DPR-style answer match: some alias is a contiguous token sequence of the passage (both from token_string)."""
    return any(alias in doc_tokens for alias in alias_tokens)


def alias_token_strings(aliases):
    return [value for value in (token_string(alias) for alias in aliases) if value.strip()]


def norm_answer(text):
    """SQuAD-style normalisation used for agreement between sampled answers."""
    words = PUNCT.sub(" ", POSSESSIVE.sub("", fold(text))).split()
    return " ".join(word for word in words if word not in {"a", "an", "the"})


def clean_answer(text):
    """First line of a closed-book generation, without an 'Answer:' prefix, quotes or a final period."""
    line = (text or "").strip().split("\n", 1)[0].strip()
    line = re.sub(r"^(answer|the answer is)\s*[:\-]?\s*", "", line, flags=re.IGNORECASE).strip()
    line = line.strip(" \"'*`").rstrip(".").strip()
    return line[:80]


def abstains(text):
    return not norm_answer(text) or bool(ABSTAIN.search(text or ""))


def correct(text, alias_norms):
    """PopQA rule: a normalised alias occurs as a whole-token substring of the normalised answer."""
    answer = f" {norm_answer(text)} "
    return answer.strip() != "" and any(f" {alias} " in answer for alias in alias_norms if alias)


def title_key(title):
    return SPACE.sub(" ", unicodedata.normalize("NFC", unquote(title or "").replace("_", " "))).casefold().strip()


def split_contents(contents):
    """wiki-18 rows store '"Title"\\ntext'; returns (title or None, text)."""
    if contents.startswith('"'):
        head, newline, text = contents.partition("\n")
        if newline:
            return head[1:-1] if head.endswith('"') and len(head) > 1 else head[1:], text
    return None, contents


def join_contents(title, text):
    return text if title is None else f'"{title}"\n{text}'


# ----------------------------------------------------------------------------- PopQA

def parse_list(value):
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        if isinstance(parsed, str):
            return [parsed]
    return [text]


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        from pyarrow.parquet import read_table as parquet

        return parquet(path).to_pylist()
    if path.suffix in {".tsv", ".csv"}:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","))
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", [])


def popqa_files(source):
    source = Path(source).resolve(strict=True)
    if source.is_file():
        return source.parent, [source]
    files = [path for path in sorted(source.rglob("*")) if path.is_file()
             and path.suffix in {".parquet", ".tsv", ".csv", ".jsonl", ".json"} and not path.name.startswith(".")
             and "train" not in path.name.lower()]
    files = [path for path in files if any(key in path.name.lower() for key in ("test", "popqa"))] or files
    if not files:
        raise FileNotFoundError(f"no PopQA table (.parquet/.tsv/.csv/.jsonl/.json) under {source}")
    return source, files


def load_popqa(source):
    base, files = popqa_files(source)
    rows, seen, dropped = [], set(), {}
    for path in files:
        for index, record in enumerate(read_table(path)):
            if any(record.get(key) in (None, "") for key in REQUIRED):
                dropped["missing_field"] = dropped.get("missing_field", 0) + 1
                continue
            question = str(record["question"]).strip()
            key = (question, str(record["obj"]))
            if key in seen:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                continue
            seen.add(key)
            aliases = []
            for alias in [str(record["obj"]), *parse_list(record["possible_answers"])]:
                alias = alias.strip()
                if len(alias) >= 2 and alias.casefold() not in {value.casefold() for value in aliases}:
                    aliases.append(alias)
            qid = str(record.get("id") if record.get("id") not in (None, "") else f"{path.stem}-{index}")
            rows.append({"qid": qid, "question": question, "prop": str(record["prop"]).strip(),
                         "subj": str(record.get("subj") or "").strip(), "obj": str(record["obj"]).strip(),
                         "s_wiki_title": str(record["s_wiki_title"]).strip(), "s_pop": number(record.get("s_pop")),
                         "o_pop": number(record.get("o_pop")), "aliases": aliases})
    if len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("PopQA ids are not unique")
    return rows, {path.relative_to(base).as_posix(): path for path in files}, dropped


# ----------------------------------------------------------------------------- spans and swaps

def whole_word(phrase):
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)


def alias_regex(aliases):
    usable = sorted({alias for alias in aliases if len(alias) >= 2}, key=lambda value: (-len(value), value))
    if not usable:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in usable) + r")(?!\w)", re.IGNORECASE)


def mentions(text, pattern):
    return [] if pattern is None else [match.span() for match in pattern.finditer(text)]


def contains(text, phrase):
    return bool(phrase) and whole_word(phrase).search(text) is not None


def replace_spans(text, spans, replacement):
    pieces, cursor = [], 0
    for start, end in sorted(spans):
        pieces += [text[cursor:start], replacement]
        cursor = end
    return "".join(pieces) + text[cursor:]


def leaks(record, title):
    """An answer alias inside the question or the passage title confounds every swap."""
    return any(contains(record["question"], alias) or (title is not None and contains(title, alias))
               for alias in record["aliases"])


def overlaps(span, spans):
    return any(span[0] < end and start < span[1] for start, end in spans)


def control_span(text, answer_spans, banned_tokens):
    """Non-answer span of the same shape as the first answer mention, replaced by the same entity in the control arm.

    Capitalised answers get a capitalised span, lower-case answers a lower-case content word. Spans touching the
    answer, the question, the title or the replacement are excluded; the closest length to the mention wins."""
    first = text[answer_spans[0][0]:answer_spans[0][1]]
    pattern = CAPITAL_SPAN if first[:1].isupper() else LOWER_WORD
    candidates = []
    for match in pattern.finditer(text):
        phrase = match.group(0).rstrip(".'-")
        if not phrase:
            continue
        words = tokens(phrase)
        if not words or set(words) & banned_tokens or all(word in CONTROL_STOP for word in words):
            continue
        occurrences = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)]
        if not occurrences or any(overlaps(span, answer_spans) for span in occurrences):
            continue
        candidates.append((abs(len(phrase) - len(first)), match.start(), phrase, occurrences))
    if not candidates:
        return None
    _, _, phrase, occurrences = min(candidates, key=lambda item: (item[0], item[1]))
    return phrase, occurrences


def related(a, b):
    """Normalised strings that are equal or contain one another as whole tokens."""
    a, b = f" {norm_answer(a)} ", f" {norm_answer(b)} "
    return a.strip() != "" and b.strip() != "" and (a in b or b in a)


def replacement_pool(rows, obscure_min=20):
    """Per relation: distinct objects; the less popular half when the relation has at least obscure_min objects."""
    by_prop = {}
    for row in rows:
        by_prop.setdefault(row["prop"], {}).setdefault(norm_answer(row["obj"]), row)
    pools = {}
    for prop, objects in by_prop.items():
        items = sorted(objects.values(), key=lambda row: (row["o_pop"] is None, row["o_pop"] or 0.0, row["obj"]))
        if len(items) >= obscure_min:
            known = [row["o_pop"] for row in items if row["o_pop"] is not None]
            median = sorted(known)[len(known) // 2] if known else None
            items = [row for row in items if median is not None and row["o_pop"] is not None and row["o_pop"] <= median]
        pools[prop] = [row["obj"] for row in items]
    return pools


def choose_replacement(record, pool, contents):
    """Deterministic same-relation entity that is not an alias and does not already occur in passage or question."""
    for candidate in sorted(pool, key=lambda value: digest("replacement", record["qid"], value)):
        if any(related(candidate, alias) for alias in record["aliases"]):
            continue
        if contains(contents, candidate) or contains(record["question"], candidate):
            continue
        return candidate
    return None


def swap_item(record, contents, replacement):
    """Treatment (answer mentions -> replacement) and control (non-answer span -> replacement) passages, or a reason."""
    title, text = split_contents(contents)
    if leaks(record, title):
        return None, "alias_in_question_or_title"
    spans = mentions(text, alias_regex(record["aliases"]))
    if not spans:
        return None, "no_alias_span"
    if replacement is None:
        return None, "no_replacement"
    banned = set(tokens(record["question"])) | set(tokens(title or "")) | set(tokens(replacement))
    for alias in record["aliases"]:
        banned |= set(tokens(alias))
    control = control_span(text, spans, banned)
    if control is None:
        return None, "no_control_span"
    phrase, occurrences = control
    return {"treat": join_contents(title, replace_spans(text, spans, replacement)),
            "control": join_contents(title, replace_spans(text, occurrences, replacement)),
            "answer_mentions": len(spans), "control_phrase": phrase, "control_mentions": len(occurrences),
            "replacement": replacement}, None


def swapped(contents, record, replacement):
    """Passage with every answer mention replaced, title untouched; None when the swap is not clean."""
    title, text = split_contents(contents)
    spans = mentions(text, alias_regex(record["aliases"]))
    if not spans or leaks(record, title) or contains(contents, replacement) or contains(record["question"], replacement):
        return None
    return join_contents(title, replace_spans(text, spans, replacement))


def pair_facts(facts):
    """Greedy pairs within (relation, stratum) in hash order; partners must have unrelated objects.

    facts: list of dicts with qid, prop, stratum, obj, aliases, question, contents. Returns [(i, j)] of fact dicts."""
    groups = {}
    for fact in facts:
        groups.setdefault((fact["prop"], fact["stratum"]), []).append(fact)
    pairs = []
    for key in sorted(groups):
        waiting = sorted(groups[key], key=lambda fact: digest("pair", fact["qid"]))
        while waiting:
            first = waiting.pop(0)
            for index, other in enumerate(waiting):
                if compatible(first, other):
                    pairs.append((first, waiting.pop(index)))
                    break
    return pairs


def compatible(a, b):
    if related(a["obj"], b["obj"]):
        return False
    if any(related(b["obj"], alias) for alias in a["aliases"]) or any(related(a["obj"], alias) for alias in b["aliases"]):
        return False
    return (swapped(a["contents"], a, b["obj"]) is not None and swapped(b["contents"], b, a["obj"]) is not None)
