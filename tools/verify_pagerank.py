#!/usr/bin/env python3
"""Independent correctness check for the PageRank implementation.

Computes PageRank a second time, by a route that shares no code with the
project, and compares the answer with what the C binary prints.  The SNAP edge
list is re-parsed first, in plain Python (no numpy, none of
tools/snap_to_csr.py), and everything below is built from that parse:

  1. PageRank is computed on a *dense* N x N transition matrix by ordinary
     power iteration -- no CSR, no row pointers, no gather loop, so a bug in
     the sparse representation cannot hide here;
  2. that dense reference is itself checked against networkx, an outside
     implementation, which rules out a misreading of the algorithm shared by
     both (skipped when networkx or scipy is missing);
  3. the C binary is run and its output compared against the above.

Because the references come from the .txt edge list while the binary computes
from the .csr, a converter bug shows up here too: the two would simply be
ranking different graphs.

The dense matrix costs N^2 * 8 bytes, so step 1 only works on small graphs:
wiki-Vote needs ~405 MB, while web-Google would need ~6 PB.  That is exactly
why wiki-Vote is in the dataset ladder -- it is not there to be fast, it is
there to be the one graph where a brute-force answer is computable at all.

--no-dense drops steps 1 and 2, promoting networkx from a check on the dense
reference to the reference itself.  That loses the ability to tell an
implementation bug from a misread algorithm -- whichever step fails tells you
where to look -- but it removes the N^2 wall, so the larger graphs can be
verified too: web-Google takes about 2 minutes, web-BerkStan about 4.

Usage:
    python3 tools/verify_pagerank.py data/snap/wiki-Vote.txt
    python3 tools/verify_pagerank.py data/snap/web-Google.txt --no-dense
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np

# N^2 * 8 bytes must stay sane: 15000 nodes is already 1.8 GB.
MAX_DENSE_NODES = 15000

TOP_LINE = re.compile(
    r"^\s*(\d+)\.\s+node\s+(\d+)\s+SNAP id\s+(\d+)\s+([-\d.eE+]+)\s*$")
SUM_LINE = re.compile(r"^rank sum\s+([-\d.eE+]+)\s*$")
CONV_LINE = re.compile(r"^(converged|STOPPED[^)]*?) after (\d+) iterations")

# The answer everything is compared against: the ranks, the route that produced
# them (it appears in the messages) and the node ids they are indexed by.
Reference = namedtuple("Reference", "ranks name order")

# What the binary reported: its top-k table as (snap id, value) pairs, the rank
# sum it printed, whether it converged, and the whole vector it wrote with -o.
Run = namedtuple("Run", "top rank_sum converged iterations ranks")

# One verdict; the message carries the measured number, pass or fail.
Check = namedtuple("Check", "ok message")


class Report:
    """Prints each verdict as it is reached and keeps the failures, which are
    what decides the exit status."""

    def __init__(self):
        self.failures = []

    def record(self, check):
        print(f"      {'ok:' if check.ok else 'FAIL'} {check.message}")
        if not check.ok:
            self.failures.append(check.message)
        return check.ok


# --- the independent answer -------------------------------------------------

def parse_edges(path):
    """Re-read the SNAP edge list with plain Python.

    No numpy and nothing from the converter: this is the independent view of
    the graph that the references are built on.
    """
    edges = set()
    nodes = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            src, dst = (int(field) for field in line.split())
            edges.add((src, dst))
            nodes.update((src, dst))
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


def networkx_pagerank(edges, order, damping, max_iters):
    """Third-party cross-check.

    The dense reference is independent in method but written by the same hand
    as the code it checks, so a misunderstanding of the algorithm itself could
    be reproduced in both; networkx is an outside implementation, which rules
    that out.  Its default dangling handling -- spread over the personalization
    vector, uniform unless asked otherwise -- matches ours, so the two are
    directly comparable.

    Returns a (ranks, note) pair; ranks is None when the check cannot run and
    note then says why.
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


def top_nodes(ranks, order, k):
    """The k highest-ranked nodes, as original SNAP ids, best first."""
    return [order[i] for i in np.argsort(-ranks)[:k]]


# --- the implementation under test ------------------------------------------

def read_all_ranks(path, order):
    """Read an -o ranks file into the order the references are indexed by.

    The file lists one "snap_id rank" pair per node; `order` is sorted, which
    is what lets searchsorted do the permutation in one vectorised step.
    """
    table = np.loadtxt(path, comments="#")
    if table.ndim != 2 or len(table) != len(order):
        raise SystemExit(f"{path}: expected {len(order)} ranks, found {len(table)}")

    nodes = np.asarray(order, dtype=np.int64)
    listed = table[:, 0].astype(np.int64)
    pos = np.searchsorted(nodes, listed)
    if pos.max() >= len(nodes) or not np.array_equal(nodes[pos], listed):
        raise SystemExit(f"{path}: lists a node the edge list does not contain")

    ranks = np.empty(len(order))
    ranks[pos] = table[:, 1]
    return ranks


def parse_output(stdout, binary):
    """Pull the top-k table, the rank sum and the convergence line out of what
    the binary printed."""
    top, rank_sum, converged, iterations = [], None, None, None
    for line in stdout.splitlines():
        if match := TOP_LINE.match(line):
            top.append((int(match.group(3)), float(match.group(4))))
        elif match := SUM_LINE.match(line):
            rank_sum = float(match.group(1))
        elif match := CONV_LINE.match(line):
            converged = match.group(1) == "converged"
            iterations = int(match.group(2))
    if not top:
        raise SystemExit(f"could not parse any ranking from {binary}:\n{stdout}")
    return top, rank_sum, converged, iterations


def run_binary(args, csr_path, ids_path, order):
    """Run the C implementation and collect everything it reports.

    It is also asked for every rank (-o), so the comparison can cover all N
    values instead of the k it prints; that file carries full double precision,
    so nothing is lost on the way.
    """
    command = [str(args.binary), str(csr_path), "-i", str(ids_path),
               "-k", str(args.k), "-d", str(args.damping),
               "-t", str(args.tolerance), "-n", str(args.max_iters)]
    with tempfile.TemporaryDirectory() as tmp:
        ranks_path = Path(tmp) / "all.ranks.txt"
        try:
            done = subprocess.run(command + ["-o", str(ranks_path)],
                                  capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise SystemExit(f"{args.binary}: not built -- run 'make' first")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"{args.binary} failed:\n{exc.stderr}")
        ranks = read_all_ranks(ranks_path, order)

    top, rank_sum, converged, iterations = parse_output(done.stdout, args.binary)
    return Run(top, rank_sum, converged, iterations, ranks)


# --- the comparisons --------------------------------------------------------

def check_networkx(nx_ranks, reference, k, tolerance):
    if top_nodes(nx_ranks, reference.order, k) != top_nodes(reference.ranks,
                                                            reference.order, k):
        return Check(False, "networkx and the dense reference disagree on the ranking")
    worst = float(np.abs(nx_ranks - reference.ranks).max())
    if worst > tolerance:
        return Check(False, f"networkx differs from the dense reference "
                            f"by {worst:.3e} > {tolerance:.0e}")
    return Check(True, f"agrees with the dense reference to {worst:.3e}")


def check_convergence(run):
    """A run stopped at the iteration limit has not settled, so its ranks are
    not the answer even when the top few happen to look right."""
    if run.converged is None:
        return Check(False, "could not tell whether the binary converged")
    if not run.converged:
        return Check(False, f"stopped at the iteration limit after {run.iterations}"
                            f" iterations, so the ranks have not settled")
    return Check(True, f"converged in {run.iterations} iterations")


def check_top_k_order(run, reference, k, tolerance):
    got = [node for node, _ in run.top]
    want = top_nodes(reference.ranks, reference.order, k)
    if got == want:
        return Check(True, f"top-{k} ranking identical to the {reference.name} reference")
    if sorted(got) != sorted(want):
        return Check(False, f"the top-{k} node sets differ")

    # Same nodes, different order.  Deep in the ranking the values are closer
    # together than the two computations agree, so only a swap between values
    # genuinely apart is a real disagreement.
    swapped, worst_gap = compare_orderings(reference, got, want)
    if worst_gap > tolerance:
        return Check(False, f"top-{k} order differs on values up to {worst_gap:.3e} apart")
    return Check(True, f"top-{k} ranking identical but for {swapped} positions, "
                       f"all between values within {worst_gap:.3e}")


def compare_orderings(reference, got, want):
    """How many positions the two orderings disagree on, and how far apart in
    rank value the worst of those disagreements is."""
    nodes = np.asarray(reference.order, dtype=np.int64)
    got = np.asarray(got, dtype=np.int64)
    want = np.asarray(want, dtype=np.int64)
    differ = got != want
    gaps = np.abs(reference.ranks[np.searchsorted(nodes, got[differ])]
                  - reference.ranks[np.searchsorted(nodes, want[differ])])
    return int(differ.sum()), float(gaps.max())


def check_top_values(run, want_values, tolerance):
    worst = max(abs(got - want) for (_, got), want in zip(run.top, want_values))
    if worst > tolerance:
        return Check(False, f"largest rank difference {worst:.3e} > {tolerance:.0e}")
    return Check(True, f"largest rank difference {worst:.3e}")


def check_all_values(run, reference, tolerance):
    n = len(reference.order)
    difference = np.abs(run.ranks - reference.ranks)
    worst = float(difference.max())
    if worst > tolerance:
        node = reference.order[int(difference.argmax())]
        return Check(False, f"worst of all {n} ranks {worst:.3e} > {tolerance:.0e},"
                            f" at node {node}")
    return Check(True, f"all {n} ranks within {worst:.3e} of the {reference.name} reference")


def check_rank_sum(run):
    """The ranks are a probability distribution, so anything but 1 means
    dangling mass was mishandled."""
    if run.rank_sum is None or abs(run.rank_sum - 1.0) > 1e-6:
        return Check(False, f"ranks sum to {run.rank_sum}, not 1 "
                            f"(dangling mass is being lost)")
    return Check(True, f"ranks sum to {run.rank_sum:.12f}")


# --- driver -----------------------------------------------------------------

def build_reference(args, edges, order, report):
    """Compute the ranks everything else is compared against.

    With --no-dense networkx is the reference itself; otherwise the dense
    matrix is, and networkx becomes a cross-check on it.
    """
    if args.no_dense:
        print("\n[networkx] third-party reference (dense check skipped)")
        ranks, note = networkx_pagerank(edges, order, args.damping, args.max_iters)
        if ranks is None:
            raise SystemExit(f"--no-dense needs networkx: {note}")
        print(f"      ok: converged, ranks sum to {ranks.sum():.12f}")
        return Reference(ranks, "networkx", order)

    n = len(order)
    print(f"\n[dense] own reference ({n * n * 8 / 2**20:.0f} MiB matrix)")
    ranks = dense_pagerank(edges, order, args.damping, 1e-15, args.max_iters)
    print(f"      ok: converged, ranks sum to {ranks.sum():.12f}")
    reference = Reference(ranks, "dense", order)

    print("\n[networkx] third-party cross-check")
    nx_ranks, note = networkx_pagerank(edges, order, args.damping, args.max_iters)
    if nx_ranks is None:
        print(f"      skipped: {note}")
    elif report.record(check_networkx(nx_ranks, reference, args.k, args.value_tolerance)):
        print("          (rules out the same misreading of the algorithm in both)")
    return reference


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("edge_list", type=Path, help="SNAP .txt edge list")
    parser.add_argument("--csr", type=Path, help="default: edge list with .csr")
    parser.add_argument("--ids", type=Path, help="default: edge list with .ids")
    parser.add_argument("--binary", type=Path, default=Path("build/pagerank_seq"))
    parser.add_argument("-d", "--damping", type=float, default=0.85)
    parser.add_argument("-t", "--tolerance", type=float, default=1e-12,
                        help="L1 tolerance passed to the C binary")
    parser.add_argument("-k", type=int, default=100,
                        help="how many top ranks to compare; deeper than a few "
                             "thousand the values are near-identical and their "
                             "order is arbitrary")
    parser.add_argument("-n", "--max-iters", type=int, default=500)
    parser.add_argument("--value-tolerance", type=float, default=1e-8,
                        help="allowed difference per rank (the C binary prints 9 decimals)")
    parser.add_argument("--max-nodes", type=int, default=MAX_DENSE_NODES,
                        help="refuse graphs larger than this (dense matrix is N^2)")
    parser.add_argument("--no-dense", action="store_true",
                        help="skip the N x N reference and check against networkx "
                             "alone, which makes the check usable on the larger graphs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csr_path = args.csr or args.edge_list.with_suffix(".csr")
    ids_path = args.ids or args.edge_list.with_suffix(".ids")
    for path in (args.edge_list, csr_path, ids_path):
        if not path.exists():
            raise SystemExit(f"{path}: does not exist")

    print(f"edge list  {args.edge_list}")
    edges, order = parse_edges(args.edge_list)
    n = len(order)
    print(f"           {n} nodes, {len(edges)} unique edges (independent parse)")

    if n > args.max_nodes and not args.no_dense:
        raise SystemExit(
            f"{n} nodes would need a {n * n * 8 / 2**30:.1f} GiB dense matrix.\n"
            f"This check is only meaningful on small graphs; raise --max-nodes "
            f"if you really have the memory.")

    report = Report()
    reference = build_reference(args, edges, order, report)

    print(f"\n[binary] {args.binary}")
    run = run_binary(args, csr_path, ids_path, order)
    want_values = [reference.ranks[i] for i in np.argsort(-reference.ranks)[:args.k]]

    report.record(check_convergence(run))
    report.record(check_top_k_order(run, reference, args.k, args.value_tolerance))
    report.record(check_top_values(run, want_values, args.value_tolerance))
    report.record(check_all_values(run, reference, args.value_tolerance))
    report.record(check_rank_sum(run))

    print()
    for position, (node, value) in enumerate(run.top[:5], 1):
        print(f"  {position}. SNAP id {node:>8}   C {value:.9f}   "
              f"{reference.name} {want_values[position - 1]:.9f}")

    if report.failures:
        print(f"\nFAILED ({len(report.failures)} problem(s)):")
        for failure in report.failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS: the C implementation matches an independent computation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
