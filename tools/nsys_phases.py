#!/usr/bin/env python3
"""Read from the nsys reports of the hybrid runs how their iterations split up.

run_benchmarks.sh runs every measurement of the hybrid build under nsys and
keeps its report as results/<machine>/nsys/<graph>-<precision>-s<share>-r<rep>
(gitignored; it opens in the Nsight Systems GUI).  This reads every report
there and writes results/<machine>/nsys_phases.csv, one row per report, that
is one row per hybrid row of bench.csv: the same run, matched on graph,
precision, share and repetition (the n-th row of a configuration in bench.csv
is its repetition n, the order in which the sweep runs them).  seconds_total
is copied from that row, so the two files can be checked against each other.

Every iteration of the hybrid build goes like this: contrib_kernel on the
GPU; the copy of contrib to the host on a second stream, while the GPU runs
the gather kernels of its own rows; the CPU waits for that copy, gathers its
rows, sends their ranks back (one copy and scatter_host_ranks) and waits for
the GPU.  The columns are the length of each of these phases, in ms, each
the median over the iterations (the first and the last left out):

  iter_ms          one iteration, from a contrib_kernel to the next
  contrib_ms       contrib_kernel
  gpu_gather_ms    the gather kernels, first start to last end
  copy_ms, copy_mb the copy of contrib to the host     (CPU rows only)
  copy_gbs         its bandwidth                        (CPU rows only)
  wait_copy_ms     the host waiting for that copy       (CPU rows only)
  cpu_gather_ms    the CPU gathering its rows           (CPU rows only)
  back_ms, back_mb the copy of the CPU's ranks back     (CPU rows only)
  scatter_ms       scatter_host_ranks                   (CPU rows only)
  wait_gpu_ms      the host waiting for the GPU at the end of the iteration

plus, once per run, outside the timed loop: upload_ms and upload_mb, the
graph going to the GPU, and download_ms, the ranks coming back.

The host phases come from the CUDA calls nsys records on the host:
cpu_gather_ms is the time between the end of the wait for the copy and the
call that sends the ranks back, which is when gather_host_rows() runs.

Usage: python3 tools/nsys_phases.py results/machine1 [--nsys /path/to/nsys]
"""

import argparse
import csv
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

COLUMNS = ["graph", "precision", "cpu_share", "rep", "seconds_total", "iterations",
           "iter_ms", "contrib_ms", "gpu_gather_ms", "copy_ms", "copy_mb", "copy_gbs",
           "wait_copy_ms", "cpu_gather_ms", "back_ms", "back_mb", "scatter_ms",
           "wait_gpu_ms", "upload_ms", "upload_mb", "download_ms"]
REPORT = re.compile(r"^(?P<graph>.+)-(?P<precision>double|float)-s(?P<share>[\d.]+)-r(?P<rep>\d+)$")
H2D, D2H = 1, 2   # copyKind in the nsys export


def export(nsys, report, sqlite_path):
    """nsys's own export of the report to SQLite, which is what gets read."""
    done = subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
                           "-o", str(sqlite_path), str(report)], capture_output=True, text=True)
    if done.returncode != 0 or not sqlite_path.exists():
        raise SystemExit(f"nsys export failed on {report}:\n{done.stderr}")


def load(sqlite_path):
    """Host CUDA calls and GPU operations of the run, joined by correlation id."""
    db = sqlite3.connect(sqlite_path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    calls = sorted((s, e, names[n].split("_v")[0], c) for s, e, n, c in db.execute(
        "SELECT start, end, nameId, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    ops = {}
    for s, e, n, c in db.execute(
            "SELECT start, end, shortName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        ops[c] = (s, e, names[n], 0)
    for s, e, k, b, c in db.execute(
            "SELECT start, end, copyKind, bytes, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        ops[c] = (s, e, {H2D: "H2D", D2H: "D2H"}.get(k, str(k)), b)
    db.close()
    return calls, ops


def ms(ns):
    return ns / 1e6


def phases(calls, ops):
    """The per-iteration medians and the one-off transfers of one run."""
    # An iteration starts where contrib_kernel is launched.
    launches = [i for i, (_, _, name, c) in enumerate(calls)
                if name == "cudaLaunchKernel" and ops.get(c, (0, 0, ""))[2] == "contrib_kernel"]
    if len(launches) < 4:
        raise SystemExit("too few iterations to take medians")
    rows = []
    for k in range(1, len(launches) - 1):          # neither the first nor the last
        window = calls[launches[k]:launches[k + 1]]
        mine = [(call, ops[call[3]]) for call in window if call[3] in ops]
        kernels = {op[2]: op for _, op in mine if op[3] == 0}
        gathers = [op for _, op in mine if op[3] == 0 and op[2].startswith("gather_")]
        row = {
            "iter_ms": ms(ops[calls[launches[k + 1]][3]][0] - kernels["contrib_kernel"][0]),
            "contrib_ms": ms(kernels["contrib_kernel"][1] - kernels["contrib_kernel"][0]),
            # none at -s 1 on a graph where every node has in-neighbours
            "gpu_gather_ms": ms(max(op[1] for op in gathers) - min(op[0] for op in gathers))
                             if gathers else 0.0,
        }
        copies = [op for _, op in mine if op[2] == "D2H"]
        sends = [(call, op) for call, op in mine if op[2] == "H2D"]
        event_sync = [call for call in window if call[2] == "cudaEventSynchronize"]
        stream_sync = [call for call in window if call[2] == "cudaStreamSynchronize"]
        if event_sync:                              # the CPU had rows
            copy = max(copies, key=lambda op: op[3])
            send_call, send = sends[0]
            row.update({
                "copy_ms": ms(copy[1] - copy[0]),
                "copy_mb": copy[3] / 1e6,
                "copy_gbs": copy[3] / (copy[1] - copy[0]),
                "wait_copy_ms": ms(event_sync[0][1] - event_sync[0][0]),
                "cpu_gather_ms": ms(send_call[0] - event_sync[0][1]),
                "back_ms": ms(send[1] - send[0]),
                "back_mb": send[3] / 1e6,
                "scatter_ms": ms(kernels["scatter_host_ranks"][1]
                                 - kernels["scatter_host_ranks"][0]),
            })
        row["wait_gpu_ms"] = ms(stream_sync[0][1] - stream_sync[0][0])
        rows.append(row)

    out = {key: statistics.median(r[key] for r in rows) for key in rows[0]}
    first, last = calls[launches[0]][0], calls[launches[-1]][0]
    uploads = [ops[c] for s, _, name, c in calls
               if s < first and name == "cudaMemcpy" and ops.get(c, (0, 0, ""))[2] == "H2D"]
    downloads = [ops[c] for s, _, name, c in calls
                 if s > last and name == "cudaMemcpy" and ops.get(c, (0, 0, ""))[2] == "D2H"]
    out["iterations"] = len(launches)
    out["upload_ms"] = ms(sum(op[1] - op[0] for op in uploads))
    out["upload_mb"] = sum(op[3] for op in uploads) / 1e6
    out["download_ms"] = ms(sum(op[1] - op[0] for op in downloads))
    return out


def bench_runs(csv_path):
    """The hybrid rows of bench.csv: (graph, precision, share) -> rows, in
    order, so that the n-th is repetition n; and the graphs in sweep order."""
    runs, order = defaultdict(list), []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["build"] != "hybrid":
                continue
            graph = Path(r["graph"]).stem
            if graph not in order:
                order.append(graph)
            runs[(graph, r["precision"], float(r["cpu_share"]))].append(r)
    return runs, order


def fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="results/<machine>, with bench.csv and nsys/")
    parser.add_argument("--nsys", default="nsys")
    args = parser.parse_args(argv)

    runs, order = bench_runs(args.results / "bench.csv")
    found = []
    for report in (args.results / "nsys").glob("*.nsys-rep"):
        m = REPORT.match(report.name[:-len(".nsys-rep")])
        if m:
            found.append((m["graph"], m["precision"], float(m["share"]), int(m["rep"]), report))
    found.sort(key=lambda f: (order.index(f[0]) if f[0] in order else len(order),
                              f[2], f[1], f[3]))

    rows, problems = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for graph, precision, share, rep, report in found:
            sqlite_path = Path(tmp) / "report.sqlite"
            export(args.nsys, report, sqlite_path)
            row = phases(*load(sqlite_path))
            same = runs.get((graph, precision, share), [])
            if rep > len(same):
                print(f"{report.name}: no repetition {rep} in bench.csv", file=sys.stderr)
                problems += 1
                seconds = ""
            else:
                seconds = same[rep - 1]["seconds_total"]
                if int(same[rep - 1]["iterations"]) != row["iterations"]:
                    print(f"{report.name}: {row['iterations']} iterations, bench.csv says "
                          f"{same[rep - 1]['iterations']}", file=sys.stderr)
                    problems += 1
            row.update(graph=graph, precision=precision, cpu_share=f"{share:g}", rep=rep,
                       seconds_total=seconds)
            rows.append({c: fmt(row[c]) if c in row else "" for c in COLUMNS})

    missing = sum(len(v) for v in runs.values()) - len(rows)
    out = args.results / "nsys_phases.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")
    if missing > 0:
        print(f"{missing} hybrid rows of bench.csv have no nsys report", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
