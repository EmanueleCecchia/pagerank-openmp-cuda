#!/usr/bin/env python3
"""Convert a SNAP directed edge list into a binary CSR file for PageRank.

PageRank pulls rank from *incoming* edges, so the CSR produced here is the
transpose of the input graph: row v lists the in-neighbours of v.  The
per-node out-degrees needed to divide each contribution are stored alongside.

SNAP input: comment lines starting with '#', then one edge per line as
"<FromNodeId>\t<ToNodeId>".  Original ids are sparse (web-Google has 875,713
nodes with ids running up to 916,428), so they are remapped to a contiguous
[0, N) range ordered by original id.

Two files are written:

  <name>.csr   binary CSR (layout below, native byte order)
  <name>.ids   uint64[N], original SNAP id of each remapped index

.csr layout:

  char[8]   magic "PRCSR001"
  uint64    n_nodes  (N)
  uint64    n_edges  (M, after de-duplication)
  uint64    row_ptr[N + 1]   in-neighbour offsets
  uint32    col_idx[M]       in-neighbours, ascending within each row
  uint32    out_deg[N]       out-degree; 0 marks a dangling node

Duplicate edges are collapsed, matching networkx.DiGraph semantics so the
kernels can be validated against networkx.pagerank on the small graphs.
"""

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np

# First eight bytes of a .csr file; must match CSR_MAGIC in src/csr.c.
# The trailing digits are a format version: bump them whenever the layout
# changes, so that old binaries reject new files instead of misreading them.
MAGIC = b"PRCSR001"
PTR_DTYPE = np.uint64  # row_ptr
IDX_DTYPE = np.uint32  # col_idx, out_deg

# Row-length bins for the adaptive CUDA granularity: one thread per row for
# short rows, one warp for medium rows, one block for the long power-law tail.
BINS = ((1, 4, "1-4  (thread/row)"), (5, 32, "5-32 (warp/row)  "), (33, None, ">32  (block/row) "))


def read_edges(path):
    """Load a SNAP edge list as an (M, 2) int64 array of original node ids."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        edges = np.loadtxt(handle, dtype=np.int64, comments="#", usecols=(0, 1))
    return edges.reshape(-1, 2)


def build_csr(edges, drop_self_loops=False):
    """Build the transposed CSR (rows = in-neighbours) plus out-degrees."""
    # Remap the sparse original ids onto [0, N).  Ravel first: the shape of
    # the inverse array for 2-D input has changed across numpy versions.
    ids, inverse = np.unique(edges.ravel(), return_inverse=True)
    inverse = inverse.reshape(-1, 2)
    src, dst = inverse[:, 0], inverse[:, 1]
    n = ids.size
    if n > np.iinfo(IDX_DTYPE).max:
        raise ValueError(f"{n} nodes exceeds the uint32 index range")

    n_self_loops = int(np.count_nonzero(src == dst))
    if drop_self_loops:
        keep = src != dst
        src, dst = src[keep], dst[keep]

    # Encoding each edge as dst * n + src lets a single np.unique both collapse
    # duplicates and sort by (dst, src) -- exactly CSR row order, with column
    # indices ascending inside every row.  Costs 8 bytes per edge; a graph with
    # billions of edges would need a C converter that sorts out of core.
    key = np.unique(dst.astype(np.int64) * n + src)
    dst, src = np.divmod(key, n)
    m = key.size

    in_deg = np.bincount(dst, minlength=n)
    row_ptr = np.zeros(n + 1, dtype=PTR_DTYPE)
    row_ptr[1:] = np.cumsum(in_deg)
    col_idx = src.astype(IDX_DTYPE)
    out_deg = np.bincount(src, minlength=n).astype(IDX_DTYPE)

    assert row_ptr[-1] == m, "row_ptr must end at the edge count"
    assert int(out_deg.sum()) == m, "out-degrees must sum to the edge count"

    dropped = n_self_loops if drop_self_loops else 0
    stats = {
        "nodes": n,
        "edges": m,
        "duplicates_removed": int(len(edges) - dropped - m),
        "self_loops": n_self_loops,
        "dangling": int(np.count_nonzero(out_deg == 0)),
        "no_in_edges": int(np.count_nonzero(in_deg == 0)),
        "max_in_deg": int(in_deg.max()),
        "max_out_deg": int(out_deg.max()),
        "in_deg": in_deg,
    }
    return ids, row_ptr, col_idx, out_deg, stats


def write_csr(path, row_ptr, col_idx, out_deg):
    with open(path, "wb") as handle:
        handle.write(MAGIC)
        np.array([row_ptr.size - 1, col_idx.size], dtype=np.uint64).tofile(handle)
        row_ptr.tofile(handle)
        col_idx.tofile(handle)
        out_deg.tofile(handle)


def report(stats):
    print(f"  nodes                {stats['nodes']:>12,}")
    print(f"  edges                {stats['edges']:>12,}")
    print(f"  duplicates removed   {stats['duplicates_removed']:>12,}")
    print(f"  self-loops           {stats['self_loops']:>12,}")
    print(f"  dangling (outdeg 0)  {stats['dangling']:>12,}")
    print(f"  no in-edges          {stats['no_in_edges']:>12,}")
    print(f"  max in / out degree  {stats['max_in_deg']:>12,} / {stats['max_out_deg']:,}")
    print("  row length distribution (in-neighbours per row):")
    in_deg = stats["in_deg"]
    for low, high, label in BINS:
        rows = (in_deg >= low) & (in_deg <= (high if high else in_deg.max()))
        count = int(np.count_nonzero(rows))
        edges = int(in_deg[rows].sum())
        print(f"    {label} {count:>12,}  ({100.0 * count / stats['nodes']:5.1f}% of rows,"
              f" {100.0 * edges / max(stats['edges'], 1):5.1f}% of edges)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="SNAP edge list (.txt or .txt.gz)")
    parser.add_argument("-o", "--output", type=Path, help="output .csr path (default: alongside input)")
    parser.add_argument("--drop-self-loops", action="store_true", help="discard u->u edges")
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"{args.input} does not exist")
    out_csr = args.output or args.input.with_suffix("").with_suffix(".csr")
    out_ids = out_csr.with_suffix(".ids")
    out_csr.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    edges = read_edges(args.input)
    parsed = time.perf_counter()
    ids, row_ptr, col_idx, out_deg, stats = build_csr(edges, args.drop_self_loops)
    built = time.perf_counter()

    write_csr(out_csr, row_ptr, col_idx, out_deg)
    ids.astype(np.uint64).tofile(out_ids)
    done = time.perf_counter()

    print(f"{args.input}  ->  {out_csr}")
    report(stats)
    size = out_csr.stat().st_size + out_ids.stat().st_size
    print(f"  on disk              {size / 2**20:>12.1f} MiB")
    print(f"  parse {parsed - start:.2f}s  build {built - parsed:.2f}s  write {done - built:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
