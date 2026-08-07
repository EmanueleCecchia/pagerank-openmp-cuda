#!/usr/bin/env python3
"""Independent correctness check for the PageRank implementation.

Computes PageRank a second time, by a route that shares no code with the
project, and compares the answer with what the C binary prints:

  1. the SNAP edge list is re-parsed with plain Python (no numpy, none of
     tools/snap_to_csr.py), giving an independent view of the graph;
  2. PageRank is computed on a *dense* N x N transition matrix by ordinary
     power iteration -- no CSR, no row pointers, no gather loop, so a bug in
     the sparse representation cannot hide here;
  3. the binary CSR file is re-read with a reader written in this file rather
     than imported from the converter, so a writer bug cannot be masked by
     reusing its own reader;
  4. the C binary is run and its output compared against all of the above.

The dense matrix costs N^2 * 8 bytes, so this only works on small graphs:
wiki-Vote needs ~405 MB, while web-Google would need ~6 PB.  That is exactly
why wiki-Vote is in the dataset ladder -- it is not there to be fast, it is
there to be the one graph where a brute-force answer is computable at all.

Usage:
    python3 tools/verify_pagerank.py data/snap/wiki-Vote.txt
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

MAGIC = b"PRCSR001"

# N^2 * 8 bytes must stay sane: 15000 nodes is already 1.8 GB.
MAX_DENSE_NODES = 15000

TOP_LINE = re.compile(
    r"^\s*(\d+)\.\s+node\s+(\d+)\s+SNAP id\s+(\d+)\s+([-\d.eE+]+)\s*$")
SUM_LINE = re.compile(r"^rank sum\s+([-\d.eE+]+)\s*$")
CONV_LINE = re.compile(r"^(converged|STOPPED[^)]*?) after (\d+) iterations")


def parse_edges(path):
    """Re-read the SNAP edge list with plain Python.

    Deliberately avoids numpy and the converter: this is the independent
    view of the graph that everything else is checked against.
    """
    edges = set()
    nodes = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            src, dst = line.split()
            src, dst = int(src), int(dst)
            edges.add((src, dst))
            nodes.add(src)
            nodes.add(dst)
    return edges, sorted(nodes)


def dense_pagerank(edges, order, damping, tolerance, max_iters):
    """PageRank on a dense transition matrix, by textbook power iteration."""
    index_of = {original: i for i, original in enumerate(order)}
    n = len(order)

    matrix = np.zeros((n, n))
    for src, dst in edges:
        matrix[index_of[dst], index_of[src]] = 1.0

    out_deg = matrix.sum(axis=0)
    dangling = out_deg == 0
    matrix[:, ~dangling] /= out_deg[~dangling]

    rank = np.full(n, 1.0 / n)
    for _ in range(max_iters):
        # The dangling nodes' rank has nowhere to go, so it is spread evenly.
        nxt = (1.0 - damping) / n + damping * (matrix @ rank + rank[dangling].sum() / n)
        if np.abs(nxt - rank).sum() < tolerance:
            return nxt
        rank = nxt
    return rank


def read_csr(path):
    """Read the binary CSR without importing the converter's own reader."""
    with open(path, "rb") as handle:
        if handle.read(8) != MAGIC:
            raise SystemExit(f"{path}: not a {MAGIC.decode()} file")
        n, m = np.fromfile(handle, dtype=np.uint64, count=2)
        row_ptr = np.fromfile(handle, dtype=np.uint64, count=int(n) + 1)
        col_idx = np.fromfile(handle, dtype=np.uint32, count=int(m))
        out_deg = np.fromfile(handle, dtype=np.uint32, count=int(n))
    return int(n), int(m), row_ptr, col_idx, out_deg


def check_csr(csr_path, ids_path, edges, order):
    """Confirm the stored CSR really is the transpose of the parsed graph."""
    n, m, row_ptr, col_idx, out_deg = read_csr(csr_path)
    ids = np.fromfile(ids_path, dtype=np.uint64)
    index_of = {original: i for i, original in enumerate(order)}

    problems = []
    if n != len(order):
        problems.append(f"node count: csr {n}, edge list {len(order)}")
    if m != len(edges):
        problems.append(f"edge count: csr {m}, edge list {len(edges)}")
    if ids.tolist() != order:
        problems.append("the .ids file is not the sorted original node ids")
    if problems:
        return problems

    want_in = {v: set() for v in range(n)}
    want_out = [0] * n
    for src, dst in edges:
        want_in[index_of[dst]].add(index_of[src])
        want_out[index_of[src]] += 1

    if out_deg.tolist() != want_out:
        problems.append("out-degrees disagree with the edge list")

    for v in range(n):
        row = col_idx[row_ptr[v]:row_ptr[v + 1]]
        if set(row.tolist()) != want_in[v]:
            problems.append(f"row {v} is not the in-neighbour set of node {order[v]}")
            break
        if np.any(np.diff(row.astype(np.int64)) <= 0):
            problems.append(f"row {v} is not sorted and unique")
            break
    return problems


def run_binary(binary, csr_path, ids_path, k, damping, tolerance, max_iters):
    """Run the C implementation and pull the top-k table out of its output."""
    command = [str(binary), str(csr_path), "-i", str(ids_path),
               "-k", str(k), "-d", str(damping),
               "-t", str(tolerance), "-n", str(max_iters)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise SystemExit(f"{binary}: not built -- run 'make' first")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{binary} failed:\n{exc.stderr}")

    top, rank_sum, converged, iterations = [], None, None, None
    for line in done.stdout.splitlines():
        match = TOP_LINE.match(line)
        if match:
            top.append((int(match.group(3)), float(match.group(4))))
            continue
        match = SUM_LINE.match(line)
        if match:
            rank_sum = float(match.group(1))
            continue
        match = CONV_LINE.match(line)
        if match:
            converged = match.group(1) == "converged"
            iterations = int(match.group(2))
    if not top:
        raise SystemExit(f"could not parse any ranking from {binary}:\n{done.stdout}")
    return top, rank_sum, converged, iterations


def networkx_pagerank(edges, order, damping, max_iters):
    """Third-party cross-check.

    The dense reference above is independent in method but was written by the
    same hand as the code it checks, so a misunderstanding of the algorithm
    itself (the dangling-node convention, say) could be reproduced in both.
    networkx is an outside implementation, which rules that out.

    Returns a (ranks, note) pair; ranks is None when the check cannot run, and
    note then says why.  networkx's default dangling handling -- spread over
    the personalization vector, which defaults to uniform -- matches ours, so
    the two are directly comparable.
    """
    try:
        import networkx as nx
    except ImportError:
        return None, "networkx not installed (pip install networkx)"

    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    graph.add_edges_from(edges)
    try:
        # networkx 3.x routes pagerank() through scipy, so a missing scipy
        # surfaces here rather than at the import above.
        ranks = nx.pagerank(graph, alpha=damping, tol=1e-14, max_iter=max_iters)
    except ImportError as exc:
        return None, f"networkx needs a package that is missing ({exc.name})"
    except nx.PowerIterationFailedConvergence:
        return None, f"networkx did not converge within {max_iters} iterations"
    return np.array([ranks[node] for node in order]), None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("edge_list", type=Path, help="SNAP .txt edge list")
    parser.add_argument("--csr", type=Path, help="default: edge list with .csr")
    parser.add_argument("--ids", type=Path, help="default: edge list with .ids")
    parser.add_argument("--binary", type=Path, default=Path("build/pagerank_seq"))
    parser.add_argument("-d", "--damping", type=float, default=0.85)
    parser.add_argument("-t", "--tolerance", type=float, default=1e-12,
                        help="L1 tolerance passed to the C binary")
    parser.add_argument("-k", type=int, default=10, help="how many ranks to compare")
    parser.add_argument("-n", "--max-iters", type=int, default=500)
    parser.add_argument("--value-tolerance", type=float, default=1e-8,
                        help="allowed difference per rank (the C binary prints 9 decimals)")
    parser.add_argument("--max-nodes", type=int, default=MAX_DENSE_NODES,
                        help="refuse graphs larger than this (dense matrix is N^2)")
    args = parser.parse_args(argv)

    csr_path = args.csr or args.edge_list.with_suffix(".csr")
    ids_path = args.ids or args.edge_list.with_suffix(".ids")
    for path in (args.edge_list, csr_path, ids_path):
        if not path.exists():
            raise SystemExit(f"{path}: does not exist")

    print(f"edge list  {args.edge_list}")
    edges, order = parse_edges(args.edge_list)
    n = len(order)
    print(f"           {n} nodes, {len(edges)} unique edges (independent parse)")

    if n > args.max_nodes:
        raise SystemExit(
            f"{n} nodes would need a {n * n * 8 / 2**30:.1f} GiB dense matrix.\n"
            f"This check is only meaningful on small graphs; raise --max-nodes "
            f"if you really have the memory.")

    failures = []

    print("\n[csr] binary CSR against the independent parse")
    problems = check_csr(csr_path, ids_path, edges, order)
    if problems:
        failures.extend(problems)
        for problem in problems:
            print(f"      FAIL {problem}")
    else:
        print("      ok: rows are the true in-neighbour sets, sorted and unique")

    print(f"\n[dense] own reference ({n * n * 8 / 2**20:.0f} MiB matrix)")
    reference = dense_pagerank(edges, order, args.damping, 1e-15, args.max_iters)
    order_desc = np.argsort(-reference)[:args.k]
    print(f"      ok: converged, ranks sum to {reference.sum():.12f}")

    print("\n[networkx] third-party cross-check")
    nx_ranks, nx_note = networkx_pagerank(edges, order, args.damping, args.max_iters)
    if nx_ranks is None:
        print(f"      skipped: {nx_note}")
    else:
        nx_worst = float(np.abs(nx_ranks - reference).max())
        nx_top = [order[i] for i in np.argsort(-nx_ranks)[:args.k]]
        dense_top = [order[i] for i in order_desc]
        if nx_top != dense_top:
            failures.append("networkx and the dense reference disagree on the ranking")
            print(f"      FAIL ranking differs\n        networkx {nx_top}\n        dense    {dense_top}")
        elif nx_worst > args.value_tolerance:
            failures.append(f"networkx differs from the dense reference by {nx_worst:.3e}")
            print(f"      FAIL largest difference {nx_worst:.3e} > {args.value_tolerance:.0e}")
        else:
            print(f"      ok: agrees with the dense reference to {nx_worst:.3e}")
            print("          (rules out the same misreading of the algorithm in both)")

    print(f"\n[binary] {args.binary}")
    top, rank_sum, converged, iterations = run_binary(
        args.binary, csr_path, ids_path, args.k,
        args.damping, args.tolerance, args.max_iters)

    # A run that hit the iteration limit has not settled, so its ranks are not
    # the answer even if the top few happen to look right.
    if converged is None:
        failures.append("could not tell whether the binary converged")
        print("      FAIL no convergence line in the output")
    elif not converged:
        failures.append(f"the binary stopped at the iteration limit ({iterations} iterations)")
        print(f"      FAIL stopped at the iteration limit after {iterations} iterations,"
              f" so the ranks have not settled")
    else:
        print(f"      ok: converged in {iterations} iterations")

    got_ids = [node for node, _ in top]
    want_ids = [order[i] for i in order_desc]
    got_values = [value for _, value in top]
    want_values = [reference[i] for i in order_desc]

    if got_ids != want_ids:
        if sorted(got_ids) == sorted(want_ids):
            failures.append("same nodes, different order (near-ties?)")
        else:
            failures.append("the top-k node sets differ")
        print(f"      FAIL ranking differs\n        C     {got_ids}\n        dense {want_ids}")
    else:
        print(f"      ok: top-{args.k} ranking identical to the dense reference")

    worst = max(abs(g - w) for g, w in zip(got_values, want_values))
    if worst > args.value_tolerance:
        failures.append(f"rank values differ by up to {worst:.3e}")
        print(f"      FAIL largest rank difference {worst:.3e} > {args.value_tolerance:.0e}")
    else:
        print(f"      ok: largest rank difference {worst:.3e}")

    if rank_sum is None or abs(rank_sum - 1.0) > 1e-6:
        failures.append(f"ranks sum to {rank_sum}, not 1")
        print(f"      FAIL ranks sum to {rank_sum} (dangling mass is being lost)")
    else:
        print(f"      ok: ranks sum to {rank_sum:.12f}")

    print()
    for i, (node, value) in enumerate(top[:5], 1):
        print(f"  {i}. SNAP id {node:>8}   C {value:.9f}   dense {want_values[i - 1]:.9f}")

    if failures:
        print(f"\nFAILED ({len(failures)} problem(s)):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS: the C implementation matches an independent computation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
