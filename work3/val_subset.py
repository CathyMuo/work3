"""Build Dr. Zero's in-training validation file data/test_1200.parquet.

The upstream solver scripts validate on ./data/test_1200.parquet, which no
upstream script creates. P0 addendum 2026-11-06: draw 1200 rows from the full
test file, stratified by data_source with proportional (largest-remainder)
allocation, numpy seed 20261106. Used only for training-time validation logs;
the P0 gate uses the full test file.

Usage: python -m work3.val_subset data/test.parquet data/test_1200.parquet
"""

from __future__ import annotations

import sys

import numpy as np

SIZE = 1200
SEED = 20261106


def allocate(counts: dict[str, int], size: int = SIZE) -> dict[str, int]:
    """Proportional allocation with largest remainders; ties broken by name."""
    total = sum(counts.values())
    if size > total:
        raise ValueError(f"cannot draw {size} from {total} rows")
    quotas = {k: size * v / total for k, v in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    left = size - sum(alloc.values())
    order = sorted(counts, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:left]:
        alloc[k] += 1
    return alloc


def subset(df, size: int = SIZE, seed: int = SEED):
    counts = df["data_source"].value_counts().to_dict()
    alloc = allocate({str(k): int(v) for k, v in counts.items()}, size)
    rng = np.random.default_rng(seed)
    picks = []
    for source in sorted(alloc):
        idx = np.flatnonzero(df["data_source"].astype(str).to_numpy() == source)
        picks.extend(sorted(rng.choice(idx, size=alloc[source], replace=False).tolist()))
    return df.iloc[sorted(picks)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    src, dst = (sys.argv[1:] if argv is None else argv)
    df = pd.read_parquet(src)
    out = subset(df)
    out.to_parquet(dst, index=False)
    print(f"wrote {len(out)} rows to {dst}")
    print(out["data_source"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
