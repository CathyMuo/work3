"""wiki-18 corpus scan, random access, query-vocabulary BM25 and the stored-FAISS parity reader.

One CPU pass (`prepare`) records line offsets, BM25 postings for PopQA query terms only, document lengths and every
passage of a PopQA subject page. Frozen protocol: research/protocols/answer_prior_p0.md.
"""

import hashlib
import json
import mmap
import os
import struct
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from work3.answer_prior_text import BM25_STOP, POSSESSIVE, TOKEN, fold, split_contents, title_key

RANGES = 64
BM25_K1, BM25_B = 0.9, 0.4


def as_numpy(values, dtype):
    return np.frombuffer(values, dtype=dtype).copy() if len(values) else np.empty(0, dtype=dtype)


def bm25_terms(text):
    return [word for word in TOKEN.findall(POSSESSIVE.sub("", fold(text))) if len(word) > 1 and word not in BM25_STOP]


def byte_ranges(path, parts=RANGES):
    """Fixed number of newline-aligned byte ranges, independent of the CPU count."""
    size = path.stat().st_size
    bounds = [0]
    with path.open("rb") as stream:
        for index in range(1, parts):
            stream.seek(max(bounds[-1], size * index // parts))
            stream.readline()
            bounds.append(min(stream.tell(), size))
    bounds.append(size)
    return [(start, end) for start, end in zip(bounds, bounds[1:]) if end > start]


def scan_range(job):
    """Scan one byte range; doc ids are local (0-based within the range) until the parent adds the base."""
    path, start, end, vocabulary, titles, out = job
    vocab = {term: index for index, term in enumerate(vocabulary)}
    titles = set(titles)
    offsets, lengths = array("q"), array("H")
    terms, docs, freqs = array("I"), array("I"), array("H")
    subject, sha = [], hashlib.sha256()
    first_id, id_mismatch, untitled, local = None, 0, 0, 0
    with open(path, "rb") as stream:
        stream.seek(start)
        position = start
        while position < end:
            line = stream.readline()
            if not line:
                break
            sha.update(line)
            offsets.append(position)
            position += len(line)
            row = json.loads(line)
            try:
                value = int(row.get("id"))
            except (TypeError, ValueError):
                value = None
            if first_id is None:
                first_id = value
            if value is None or first_id is None or value != first_id + local:
                id_mismatch += 1
            contents = row["contents"]
            title, _ = split_contents(contents)
            if title is None:
                untitled += 1
            elif title_key(title) in titles:
                subject.append((local, contents))
            words = bm25_terms(contents)
            lengths.append(min(len(words), 65535))
            counts = {}
            for word in words:
                if word in vocab:
                    counts[word] = counts.get(word, 0) + 1
            for word, count in counts.items():
                terms.append(vocab[word])
                docs.append(local)
                freqs.append(min(count, 65535))
            local += 1
    np.savez(out, offsets=as_numpy(offsets, np.int64), lengths=as_numpy(lengths, np.uint16),
             terms=as_numpy(terms, np.uint32), docs=as_numpy(docs, np.uint32), freqs=as_numpy(freqs, np.uint16))
    return {"start": start, "end": end, "rows": local, "first_id": first_id, "id_mismatch": id_mismatch,
            "untitled": untitled, "sha256": sha.hexdigest(), "subject": subject, "out": str(out)}


def scan_corpus(path, vocabulary, titles, directory, processes, progress=None):
    """Parallel scan; writes offsets.npy, lengths.npy, postings_*.npy and subject_docs.jsonl into directory."""
    path = Path(path).resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(str(path), start, end, list(vocabulary), sorted(titles), directory / f"range-{index:03d}.npz")
            for index, (start, end) in enumerate(byte_ranges(path))]
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, processes)) as pool:
        futures = {pool.submit(scan_range, job): index for index, job in enumerate(jobs)}
        done = 0
        for future in futures:
            results[futures[future]] = future.result()
            done += 1
            if progress:
                progress(done, len(jobs), sum(result["rows"] for result in results if result))
    base, bases = 0, []
    for result in results:
        bases.append(base)
        base += result["rows"]
    rows = base
    offsets = np.empty(rows + 1, dtype=np.int64)
    lengths = np.empty(rows, dtype=np.uint16)
    parts = {"terms": [], "docs": [], "freqs": []}
    id_mismatch = 0
    with (directory / "subject_docs.jsonl").open("w", encoding="utf-8") as subject:
        for base, result in zip(bases, results):
            with np.load(result["out"]) as data:
                offsets[base:base + result["rows"]] = data["offsets"]
                lengths[base:base + result["rows"]] = data["lengths"]
                parts["terms"].append(data["terms"])
                parts["docs"].append(data["docs"].astype(np.uint32) + np.uint32(base))
                parts["freqs"].append(data["freqs"])
            # rows whose "id" field differs from the global line number (the index row order)
            id_mismatch += result["id_mismatch"] if result["first_id"] == base else result["rows"]
            for local, contents in result["subject"]:
                subject.write(json.dumps({"doc": base + local, "contents": contents}, ensure_ascii=False) + "\n")
            os.unlink(result["out"])
    offsets[rows] = path.stat().st_size
    np.save(directory / "offsets.npy", offsets)
    np.save(directory / "lengths.npy", lengths)
    for name, values in parts.items():
        dtype = {"terms": np.uint32, "docs": np.uint32, "freqs": np.uint16}[name]
        np.save(directory / f"postings_{name}.npy", np.concatenate(values).astype(dtype) if values else
                np.empty(0, dtype=dtype))
    combined = hashlib.sha256("".join(result["sha256"] for result in results).encode()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "rows": rows, "ranges": len(results),
            "range_sha256": combined, "id_mismatch": id_mismatch,
            "untitled": sum(result["untitled"] for result in results),
            "subject_docs": sum(len(result["subject"]) for result in results),
            "postings": int(sum(len(values) for values in parts["terms"]))}


class Corpus:
    """Random access to wiki-18 rows by line number through the offsets from prepare."""

    def __init__(self, path, offsets):
        self.stream = open(path, "rb")
        self.map = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        self.offsets = np.load(offsets, mmap_mode="r")

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, doc):
        start, end = int(self.offsets[doc]), int(self.offsets[doc + 1])
        return json.loads(self.map[start:end])["contents"]

    def many(self, docs):
        """Contents for many ids, read in file order."""
        unique = sorted({int(doc) for doc in docs})
        return {doc: self.get(doc) for doc in unique}

    def close(self):
        self.map.close()
        self.stream.close()


# ----------------------------------------------------------------------------- BM25

class BM25:
    """Lucene-style BM25 (k1=0.9, b=0.4) restricted to the PopQA query vocabulary; exact for those queries."""

    def __init__(self, directory, vocabulary):
        directory = Path(directory)
        lengths = np.load(directory / "lengths.npy").astype(np.float32)
        terms = np.load(directory / "postings_terms.npy")
        docs = np.load(directory / "postings_docs.npy")
        freqs = np.load(directory / "postings_freqs.npy").astype(np.float32)
        self.rows = len(lengths)
        order = np.argsort(terms, kind="stable")
        terms, self.docs, freqs = terms[order], docs[order], freqs[order]
        self.vocab = {term: index for index, term in enumerate(vocabulary)}
        self.pointer = np.searchsorted(terms, np.arange(len(vocabulary) + 1)).astype(np.int64)
        df = np.diff(self.pointer).astype(np.float64)
        self.idf = np.log1p((self.rows - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths / max(float(lengths.mean()), 1e-9))
        self.weight = (np.repeat(self.idf, np.diff(self.pointer)) * freqs * (BM25_K1 + 1)
                       / (freqs + norm[self.docs])).astype(np.float32)
        self.accumulator = np.zeros(self.rows, dtype=np.float32)

    def search(self, query, k):
        counts = {}
        for word in bm25_terms(query):
            if word in self.vocab:
                counts[self.vocab[word]] = counts.get(self.vocab[word], 0) + 1
        touched = []
        for term, count in counts.items():
            start, end = self.pointer[term], self.pointer[term + 1]
            if end > start:
                docs = self.docs[start:end]
                self.accumulator[docs] += count * self.weight[start:end]
                touched.append(docs)
        if not touched:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        candidates = np.unique(np.concatenate(touched))
        scores = self.accumulator[candidates]
        self.accumulator[candidates] = 0.0
        top = np.argsort(-scores, kind="stable")[:k]
        return candidates[top].astype(np.int64), scores[top]


def rrf(rankings, k, constant=60):
    """Reciprocal-rank fusion of several ranked id lists; ties broken by the first list's order."""
    scores, first = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc = int(doc)
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (constant + rank + 1)
            first.setdefault(doc, len(first))
    return [doc for doc, _ in sorted(scores.items(), key=lambda item: (-item[1], first[item[0]]))[:k]]


# ----------------------------------------------------------------------------- stored FAISS flat index (parity only)

def flat_index_vectors(path):
    """Memory-map the float32 vectors of a FAISS IndexFlat file, validating every header field; None if unparsable."""
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(64)
    if len(head) < 45 or head[:4] not in (b"IxFI", b"IxF2"):
        return None
    dim, total = struct.unpack_from("<iq", head, 4)
    metric = struct.unpack_from("<i", head, 4 + 4 + 8 + 8 + 8 + 1)[0]
    offset = 4 + 4 + 8 + 8 + 8 + 1 + 4 + (4 if metric > 1 else 0)
    count = struct.unpack_from("<Q", head, offset)[0]
    offset += 8
    if dim <= 0 or total <= 0 or count != dim * total or path.stat().st_size != offset + 4 * count:
        return None
    return np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(total, dim))
