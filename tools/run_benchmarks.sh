#!/usr/bin/env bash
# Runs the full experiment sweep and collects the results under results/.
#
#   results/bench.csv        one row per run (graph, build, threads, timing)
#   results/<graph>.ranks.txt  the ranks themselves, from the OpenMP build
#
# Every configuration is run REPS times because a single timing on a laptop is
# noise; take the minimum per configuration when building the tables, since
# the fastest run is the one least disturbed by other activity.
#
# Override any of the settings from the environment, e.g.
#   GRAPHS="wiki-Vote web-Google" REPS=1 tools/run_benchmarks.sh

set -euo pipefail

GRAPHS=${GRAPHS:-"wiki-Vote web-NotreDame web-Stanford web-Google web-BerkStan cit-Patents wiki-topcats soc-Pokec soc-LiveJournal1"}
THREADS=${THREADS:-"1 2 4 8"}
REPS=${REPS:-3}
DATA=${DATA:-data/snap}
OUT=${OUT:-results}

cd "$(dirname "$0")/.."

if [ ! -x build/pagerank_omp ]; then
    echo "build/pagerank_omp missing -- run 'make' first" >&2
    exit 1
fi

mkdir -p "$OUT"
rm -f "$OUT/bench.csv"

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
