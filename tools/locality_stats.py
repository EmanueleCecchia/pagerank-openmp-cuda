#!/usr/bin/env python3
"""Measure the locality of the contrib[col_idx[j]] gather over .csr graphs.

The kernel reads contrib out of order, one element per edge.  What that read
costs does not depend on how many edges there are but on how many distinct
cache lines are needed to serve them: when a node's predecessors have nearby
indices, one 64-byte line covers several of them at once.

Distinct lines are counted over blocks of 256 consecutive rows, which is the
scheduling unit of the loop (schedule(dynamic, 256) in pagerank.c): inside a
block the reuse is captured by the L1-L2 caches, across blocks it is not.

Usage: python3 tools/locality_stats.py data/snap/*.csr
"""

import sys
from pathlib import Path

import numpy as np

MAGIC = b"PRCSR001"
LINEA = 64          # bytes in a cache line
BLOCCO = 256        # rows per chunk, matching schedule(dynamic, 256)


def carica(path):
    """Read a .csr file; see tools/snap_to_csr.py for the format."""
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: non e' un file .csr")
        n, m = (int(x) for x in np.fromfile(fh, dtype=np.uint64, count=2))
        row_ptr = np.fromfile(fh, dtype=np.uint64, count=n + 1)
        col_idx = np.fromfile(fh, dtype=np.uint32, count=m)
        out_deg = np.fromfile(fh, dtype=np.uint32, count=n)
    return n, m, row_ptr, col_idx, out_deg


def salto_mediano(row_ptr, col_idx, deg):
    """Median of col_idx[j+1] - col_idx[j] between edges of the same row.

    Indices ascend inside every row, so the gap is always positive; the steps
    that cross from one row into the next are discarded, since they are the
    start of a new row rather than an access the loop could do anything about.
    """
    salti = np.diff(col_idx.astype(np.int64))
    interni = np.ones(salti.size, dtype=bool)
    inizi = row_ptr[:-1].astype(np.int64)
    interni[inizi[1:][deg[1:] > 0] - 1] = False
    return float(np.median(salti[interni]))


def linee_per_arco(row_ptr, col_idx, n, m, taglia_elemento):
    """Distinct cache lines touched, summed over the blocks of 256 rows."""
    per_linea = LINEA // taglia_elemento
    linee = (col_idx >> int(np.log2(per_linea))).astype(np.int64)
    distinte = 0
    for c in range(0, n, BLOCCO):
        a, b = int(row_ptr[c]), int(row_ptr[min(c + BLOCCO, n)])
        if b > a:
            distinte += np.unique(linee[a:b]).size
    return distinte / m


def main():
    percorsi = sys.argv[1:]
    if not percorsi:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 1

    print(f"{'grafo':<18}{'N':>10}{'M':>12}{'riga':>7}"
          f"{'contrib':>10}{'salto':>9}{'linee/arco':>12}{'archi/linea':>13}")
    for p in percorsi:
        n, m, row_ptr, col_idx, _ = carica(p)
        deg = np.diff(row_ptr).astype(np.int64)
        lpa = linee_per_arco(row_ptr, col_idx, n, m, taglia_elemento=8)
        print(f"{Path(p).stem:<18}{n:>10,}{m:>12,}{m/n:>7.2f}"
              f"{n*8/2**20:>9.1f}M{salto_mediano(row_ptr, col_idx, deg):>9,.0f}"
              f"{lpa:>12.3f}{1/lpa:>13.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
