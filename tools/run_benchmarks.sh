#!/usr/bin/env bash
# Runs the full experiment sweep on this machine and collects the results
# under results/<machine>/, one folder per machine measured:
#
#   results/<machine>/bench.csv          one row per run (graph, build, threads, timing)
#   results/<machine>/<graph>.ranks.txt  the ranks themselves, from the OpenMP build
#
# The machine name is required, so that a sweep never overwrites the results
# of another machine by accident:
#   MACHINE=machine2 tools/run_benchmarks.sh
#
# Every configuration is run REPS times because a single timing on a laptop is
# noise; take the minimum per configuration when building the tables, since
# the fastest run is the one least disturbed by other activity.
#
# Override any of the other settings from the environment, e.g.
#   MACHINE=machine2 GRAPHS="wiki-Vote web-Google" REPS=1 tools/run_benchmarks.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${MACHINE:-}" ]; then
    echo "set MACHINE to name this machine's results, e.g." >&2
    echo "  MACHINE=machine2 tools/run_benchmarks.sh" >&2
    echo "already measured: $(cd results 2>/dev/null && ls -d */ 2>/dev/null | tr -d / | xargs)" >&2
    exit 1
fi

# Powers of two up to the logical CPUs, plus the physical cores and the logical
# CPUs themselves: 1 2 4 8 on a 4-core/8-thread laptop, 1 2 4 8 12 16 24 on a
# 12-core/24-thread machine.
default_threads() {
    local logical physical p=1 list=""
    logical=$(nproc)
    physical=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
    while [ "$p" -le "$logical" ]; do
        list="$list $p"
        p=$((p * 2))
    done
    echo $list "$physical" "$logical" | tr ' ' '\n' | sort -nu | xargs
}

GRAPHS=${GRAPHS:-"wiki-Vote web-NotreDame web-Stanford web-Google web-BerkStan cit-Patents wiki-topcats soc-Pokec soc-LiveJournal1"}
THREADS=${THREADS:-$(default_threads)}
REPS=${REPS:-3}
DATA=${DATA:-data/snap}
OUT=${OUT:-results/$MACHINE}

if [ ! -x build/pagerank_omp ]; then
    echo "build/pagerank_omp missing -- run 'make' first" >&2
    exit 1
fi

mkdir -p "$OUT"
rm -f "$OUT/bench.csv"

echo "machine $MACHINE, threads: $THREADS"
for graph in $GRAPHS; do
    csr="$DATA/$graph.csr"
    ids="$DATA/$graph.ids"

    if [ ! -f "$csr" ]; then
        echo "skipping $graph: $csr not found (run tools/snap_to_csr.py first)" >&2
        continue
    fi

    echo "== $graph"
    for rep in $(seq "$REPS"); do
        # Sequential build: genuinely no OpenMP, not one thread of it.
        ./build/pagerank_seq "$csr" -c "$OUT/bench.csv" >/dev/null
        for t in $THREADS; do
            OMP_NUM_THREADS=$t ./build/pagerank_omp "$csr" -c "$OUT/bench.csv" >/dev/null
        done
        ./build/pagerank_omp_float "$csr" -c "$OUT/bench.csv" >/dev/null
        echo "   repetition $rep done"
    done

    # The ranks themselves only need computing once.
    ./build/pagerank_omp "$csr" -i "$ids" -o "$OUT/$graph.ranks.txt" >/dev/null
done

echo
echo "wrote $OUT/bench.csv ($(( $(wc -l < "$OUT/bench.csv") - 1 )) runs)"
ls -1 "$OUT"/*.ranks.txt 2>/dev/null | while read -r f; do
    echo "wrote $f ($(du -h "$f" | cut -f1))"
done
