#!/usr/bin/env python3
"""Show experimentally that the gather cost comes from the node numbering.

Relabels the nodes of a .csr graph at random and measures again.  The
permutation leaves the node count, the edge count and every single row length
untouched, and therefore the whole of Table tab:bins as well: the only thing
it destroys is the locality of the indices read from contrib.  If the times
go up, the gap between web-Google and web-BerkStan is a matter of numbering
and not of row shape.

Usage: python3 tools/locality_experiment.py data/snap/web-BerkStan.csr
       python3 tools/locality_experiment.py data/snap/web-BerkStan.csr -p 1 4 8 -r 5
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"PRCSR001"
SEED = 12345


def read_csr(path):
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: not a .csr file")
        n, m = (int(x) for x in np.fromfile(fh, dtype=np.uint64, count=2))
        row_ptr = np.fromfile(fh, dtype=np.uint64, count=n + 1)
        col_idx = np.fromfile(fh, dtype=np.uint32, count=m)
        out_deg = np.fromfile(fh, dtype=np.uint32, count=n)
    return n, m, row_ptr, col_idx, out_deg


def write_csr(path, row_ptr, col_idx, out_deg):
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        np.array([row_ptr.size - 1, col_idx.size], dtype=np.uint64).tofile(fh)
        row_ptr.astype(np.uint64).tofile(fh)
        col_idx.astype(np.uint32).tofile(fh)
        out_deg.astype(np.uint32).tofile(fh)


def relabel(n, m, row_ptr, col_idx, out_deg):
    """Relabel the nodes at random, keeping each row sorted afterwards."""
    rng = np.random.default_rng(SEED)
    label = rng.permutation(n).astype(np.uint32)      # label[v] = new label of v
    order = np.argsort(label)                         # order[k] = old node now labelled k
    deg = np.diff(row_ptr).astype(np.int64)

    new_deg = deg[order]
    new_row_ptr = np.zeros(n + 1, dtype=np.uint64)
    new_row_ptr[1:] = np.cumsum(new_deg)

    start = row_ptr[:-1].astype(np.int64)[order]
    pick = np.repeat(start - new_row_ptr[:-1].astype(np.int64), new_deg) + np.arange(m)
    new_col_idx = label[col_idx[pick]]

    # col_idx must stay ascending inside every row, the way the converter writes it
    row = np.repeat(np.arange(n, dtype=np.int64), new_deg)
    new_col_idx = new_col_idx[np.lexsort((new_col_idx, row))]

    assert np.array_equal(np.sort(new_deg), np.sort(deg)), "row lengths altered"
    return new_row_ptr, new_col_idx, out_deg[order]


def measure(binary, graph, threads, repeats, csv_tmp):
    """Minimum over several runs: system activity can only ever slow things down."""
    best = None
    for _ in range(repeats):
        if csv_tmp.exists():
            csv_tmp.unlink()
        res = subprocess.run([str(binary), str(graph), "-c", str(csv_tmp)],
                             capture_output=True, text=True,
                             env={**os.environ, "OMP_NUM_THREADS": str(threads)})
        if res.returncode != 0:
            sys.exit(f"{binary} exited with {res.returncode}:\n{res.stderr}")
        with open(csv_tmp) as fh:
            r = list(csv.DictReader(fh))[-1]
        t = float(r["seconds_total"])
        if best is None or t < best[0]:
            best = (t, int(r["iterations"]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("graph", type=Path)
    ap.add_argument("-b", "--binary", type=Path, default=None,
                    help="default: build/pagerank_omp (or .exe on Windows)")
    ap.add_argument("-p", "--threads", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("-r", "--repeats", type=int, default=3)
    args = ap.parse_args()

    binary = args.binary
    if binary is None:
        for c in (Path("build/pagerank_omp"), Path("build/pagerank_omp.exe")):
            if c.exists():
                binary = c
                break
        else:
            sys.exit("binary not found: run make, or pass -b")

    n, m, row_ptr, col_idx, out_deg = read_csr(args.graph)
    print(f"{args.graph.stem}: N={n:,}  M={m:,}  binary={binary}")
    print("relabelling the nodes at random (N, M and every row length stay the same)")

    tmp = Path(tempfile.mkdtemp(prefix="locality-"))
    try:
        shuffled = tmp / (args.graph.stem + "-shuffled.csr")
        write_csr(shuffled, *relabel(n, m, row_ptr, col_idx, out_deg))
        csv_tmp = tmp / "run.csv"

        print(f"\n{'threads':>8}{'original':>12}{'shuffled':>12}{'change':>9}"
              f"{'ns/edge orig.':>15}{'ns/edge shuf.':>15}")
        for p in args.threads:
            t0, it0 = measure(binary, args.graph, p, args.repeats, csv_tmp)
            t1, it1 = measure(binary, shuffled, p, args.repeats, csv_tmp)
            if it0 != it1:
                print(f"  warning: iteration counts differ ({it0} against {it1})")
            print(f"{p:>8}{t0:>11.3f}s{t1:>11.3f}s{(t1/t0 - 1)*100:>8.0f}%"
                  f"{t0/it0/m*1e9:>15.2f}{t1/it1/m*1e9:>15.2f}")
        print(f"\nminimum over {args.repeats} runs per configuration")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
