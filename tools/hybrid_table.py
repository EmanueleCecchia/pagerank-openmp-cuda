#!/usr/bin/env python3
"""Print the times of the hybrid build, one row per graph, one column per share.

For every machine given, reads its bench.csv (written by run_benchmarks.sh)
and prints, for each graph the hybrid build ran on, the time at every share
of the edges for the CPU that was measured (-s; 0 is the GPU alone), each the
minimum over the repetitions.  The last column is the best OpenMP time on the
same machine, at whatever thread count gave it, for reference.  Every share
measured gets its column: none is singled out.

One table per precision the hybrid build was measured in.

Usage: python3 tools/hybrid_table.py results/machine1 [results/machine3 ...]
"""

import csv
import sys
from pathlib import Path


def load(path):
    """Minimum seconds per (graph, build, precision, threads, cpu_share), and
    the graphs in the order the sweep ran them."""
    if path.is_dir():
        path = path / "bench.csv"
    best, graphs = {}, []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            graph = Path(r["graph"]).stem
            if graph not in graphs:
                graphs.append(graph)
            key = (graph, r["build"], r["precision"], int(r["threads"]),
                   float(r["cpu_share"]))
            best[key] = min(best.get(key, float("inf")), float(r["seconds_total"]))
    return best, graphs


def print_tables(name, best, graphs):
    if not any(b == "hybrid" for (_, b, _, _, _) in best):
        print(f"\n{name}: no runs of the hybrid build yet")
        return
    for precision in ("double", "float"):
        times = {(g, s): t for (g, b, p, _, s), t in best.items()
                 if b == "hybrid" and p == precision}
        if not times:
            continue
        shares = sorted({s for _, s in times})
        print(f"\n{name}, hybrid build, {precision}: seconds per share of the edges "
              f"for the CPU (0 = GPU alone)")
        print(f"{'graph':<18}" + "".join(f"{s:>9g}" for s in shares) + f"{'best omp':>16}")
        for g in graphs:
            if not any((g, s) in times for s in shares):
                continue
            cells = "".join(f"{times[(g, s)]:>9.4f}" if (g, s) in times else f"{'-':>9}"
                            for s in shares)
            omp = [(t, th) for (gg, b, p, th, _), t in best.items()
                   if gg == g and b == "omp" and p == precision]
            ref = f"{min(omp)[0]:>9.4f} ({min(omp)[1]:>2}t)" if omp else f"{'-':>16}"
            print(f"{g:<18}{cells}{ref}")


def main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 1
    for arg in argv:
        path = Path(arg)
        best, graphs = load(path)
        print_tables(path.name if path.is_dir() else path.parent.name, best, graphs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
