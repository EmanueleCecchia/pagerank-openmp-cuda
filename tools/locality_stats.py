#!/usr/bin/env python3
"""Measure the locality of the contrib[col_idx[j]] gather over .csr graphs.

The kernel reads contrib out of order, one element per edge.  What that read
costs does not depend on how many edges there are but on how many distinct
cache lines are needed to serve them: when a node's predecessors have nearby
indices, one 64-byte line covers several of them at once.

Distinct lines are counted over chunks of 256 consecutive rows, which is the
scheduling unit of the loop (schedule(dynamic, 256) in pagerank.c): inside a
chunk the reuse is captured by the L1-L2 caches, across chunks it is not.

Usage: python3 tools/locality_stats.py data/snap/*.csr
"""

import sys
from pathlib import Path

import numpy as np

MAGIC = b"PRCSR001"
LINE = 64           # bytes in a cache line
CHUNK = 256         # rows per chunk, matching schedule(dynamic, 256)


def read_csr(path):
    """Read a .csr file; see tools/snap_to_csr.py for the format."""
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a .csr file")
        n, m = (int(x) for x in np.fromfile(fh, dtype=np.uint64, count=2))
        row_ptr = np.fromfile(fh, dtype=np.uint64, count=n + 1)
        col_idx = np.fromfile(fh, dtype=np.uint32, count=m)
        out_deg = np.fromfile(fh, dtype=np.uint32, count=n)
    return n, m, row_ptr, col_idx, out_deg


def median_gap(row_ptr, col_idx, deg):
    """Median of col_idx[j+1] - col_idx[j] between edges of the same row.

    Indices ascend inside every row, so the gap is always positive; the steps
    that cross from one row into the next are discarded, since they are the
    start of a new row rather than an access the loop could do anything about.
    """
    gaps = np.diff(col_idx.astype(np.int64))
    inside = np.ones(gaps.size, dtype=bool)
    starts = row_ptr[:-1].astype(np.int64)
    inside[starts[1:][deg[1:] > 0] - 1] = False
    return float(np.median(gaps[inside]))


def lines_per_edge(row_ptr, col_idx, n, m, element_size):
    """Distinct cache lines touched, summed over the chunks of 256 rows."""
    per_line = LINE // element_size
    lines = (col_idx >> int(np.log2(per_line))).astype(np.int64)
    distinct = 0
    for c in range(0, n, CHUNK):
        a, b = int(row_ptr[c]), int(row_ptr[min(c + CHUNK, n)])
        if b > a:
            distinct += np.unique(lines[a:b]).size
    return distinct / m


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 1

    print(f"{'graph':<18}{'N':>10}{'M':>12}{'row':>7}"
          f"{'contrib':>10}{'gap':>9}{'lines/edge':>12}{'edges/line':>13}")
    for p in paths:
        n, m, row_ptr, col_idx, _ = read_csr(p)
        deg = np.diff(row_ptr).astype(np.int64)
        lpe = lines_per_edge(row_ptr, col_idx, n, m, element_size=8)
        print(f"{Path(p).stem:<18}{n:>10,}{m:>12,}{m/n:>7.2f}"
              f"{n*8/2**20:>9.1f}M{median_gap(row_ptr, col_idx, deg):>9,.0f}"
              f"{lpe:>12.3f}{1/lpe:>13.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
