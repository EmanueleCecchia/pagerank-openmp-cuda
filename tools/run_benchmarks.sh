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
# It may name a sub-folder too: on a machine with more than one GPU the CPU
# builds go in the machine's folder, and the hybrid one in a folder per GPU,
# picked with CUDA_VISIBLE_DEVICES:
#   MACHINE=machine3 BUILDS="seq omp" tools/run_benchmarks.sh
#   CUDA_VISIBLE_DEVICES=0 MACHINE=machine3/x16 BUILDS=hybrid tools/run_benchmarks.sh
#
# Every configuration is run REPS times because a single timing on a laptop is
# noise; take the minimum per configuration when building the tables, since
# the fastest run is the one least disturbed by other activity.
#
# BUILDS picks what to measure: seq, omp (with its float variant) and hybrid
# (with its float variant, at every share of the edges in SHARES).  A sweep
# replaces in bench.csv only the rows of the graphs, builds and, for the
# hybrid one, shares it measures, so the hybrid build can be measured without
# touching the CPU results, vice versa, and one share without the others:
#   MACHINE=machine1 BUILDS=hybrid tools/run_benchmarks.sh
#   MACHINE=machine1 BUILDS=hybrid SHARES=1 tools/run_benchmarks.sh
#
# When nsys is installed every run of the hybrid build goes under it, and its
# report is kept in results/<machine>/nsys/ (gitignored): at the end
# tools/nsys_phases.py reads from each report how that run's iterations split
# up into results/<machine>/nsys_phases.csv, one row for every hybrid row of
# bench.csv.  NSYS= turns it off; NSYS=/path/to/nsys picks another one.
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
BUILDS=${BUILDS:-"seq omp hybrid"}
THREADS=${THREADS:-$(default_threads)}
# Shares of the edges for the CPU in the hybrid build, each one measured and
# recorded: 0 is the GPU alone, 1 the CPU alone but for contrib_kernel and
# the rows with no in-neighbours.
SHARES=${SHARES:-"0 0.05 0.1 0.25 0.5 0.75 1"}
REPS=${REPS:-3}
DATA=${DATA:-data/snap}
OUT=${OUT:-results/$MACHINE}
CSV="$OUT/bench.csv"
NSYS=${NSYS-$(command -v nsys || true)}
NSYS_DIR="$OUT/nsys"
PYTHON=${PYTHON:-python3}

measures() {
    case " $BUILDS " in *" $1 "*) return 0 ;; esac
    return 1
}

for build in $BUILDS; do
    case $build in
        seq)    needed="pagerank_seq" ;;
        omp)    needed="pagerank_omp pagerank_omp_float" ;;
        hybrid) needed="pagerank_hybrid pagerank_hybrid_float" ;;
        *)      echo "unknown build '$build' in BUILDS (seq, omp, hybrid)" >&2; exit 1 ;;
    esac
    for exe in $needed; do
        if [ ! -x "build/$exe" ]; then
            echo "build/$exe missing -- run 'make' first" >&2
            exit 1
        fi
    done
done

mkdir -p "$OUT"

# Drop the rows this sweep is about to measure again, keep every other one.
# The graph column holds the .csr path, so the graph is its last component;
# the hybrid rows also match on the share, column 14, compared as numbers so
# that 0.50 is 0.5.  In the C locale: with a decimal comma in LC_NUMERIC
# (es_ES, it_IT) mawk reads 0.25 as 0, every share below 1 matches 0, and
# measuring 0.05 alone would drop the rows of 0, 0.25, 0.5 and 0.75 too.
if [ -f "$CSV" ]; then
    LC_ALL=C awk -F, -v graphs="$GRAPHS" -v builds="$BUILDS" -v shares="$SHARES" '
        BEGIN { split(graphs, g, " "); for (i in g) G[g[i] ".csr"] = 1
                split(builds, b, " "); for (i in b) B[b[i]] = 1
                split(shares, s, " "); for (i in s) S[s[i] + 0] = 1 }
        NR == 1 { print; next }
        { n = split($1, path, "/")
          again = (path[n] in G) && ($4 in B) && ($4 != "hybrid" || ($14 + 0) in S)
          if (!again) print }
    ' "$CSV" > "$CSV.tmp"
    mv "$CSV.tmp" "$CSV"
fi
# The same for the nsys reports of the hybrid runs measured again.
if measures hybrid && [ -n "$NSYS" ]; then
    mkdir -p "$NSYS_DIR"
    for graph in $GRAPHS; do
        for s in $SHARES; do
            rm -f "$NSYS_DIR/$graph"-double-s"$s"-r*.nsys-rep "$NSYS_DIR/$graph"-float-s"$s"-r*.nsys-rep
        done
    done
fi

# One run of the hybrid build, on every logical CPU for the CPU's rows, as the
# OpenMP build at its largest thread count.  Under nsys, if any, tracing the
# CUDA calls only: sampling the CPU would slow down the runs where the CPU
# does most of the work.  The report is the very run whose time goes in
# bench.csv.
# --show-output=false keeps the program's output in the report only: to echo
# it on the console nsys 2022-2023 starts nsys-tee, which spins on a CPU of
# its own, and with every logical CPU running an OpenMP thread the barrier of
# the CPU's gather waits for the thread left without one -- on Machine 2 that
# added ~0.1 s to every run with a CPU share.  The output goes to /dev/null
# here anyway.
run_hybrid() {    # executable precision share repetition
    if [ -n "$NSYS" ]; then
        OMP_NUM_THREADS=$(nproc) "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
            --show-output=false \
            --force-overwrite true -o "$NSYS_DIR/$graph-$2-s$3-r$4" \
            "$1" "$csr" -s "$3" -c "$CSV" >/dev/null
    else
        OMP_NUM_THREADS=$(nproc) "$1" "$csr" -s "$3" -c "$CSV" >/dev/null
    fi
}

echo "machine $MACHINE, builds: $BUILDS"
measures omp && echo "OpenMP threads: $THREADS"
measures hybrid && echo "hybrid shares: $SHARES, on $(nproc) OpenMP threads${NSYS:+, under nsys}"
for graph in $GRAPHS; do
    csr="$DATA/$graph.csr"
    ids="$DATA/$graph.ids"

    if [ ! -f "$csr" ]; then
        echo "skipping $graph: $csr not found (run tools/snap_to_csr.py first)" >&2
        continue
    fi

    echo "== $graph"
    for rep in $(seq "$REPS"); do
        if measures seq; then
            # Sequential build: genuinely no OpenMP, not one thread of it.
            ./build/pagerank_seq "$csr" -c "$CSV" >/dev/null
        fi
        if measures omp; then
            for t in $THREADS; do
                OMP_NUM_THREADS=$t ./build/pagerank_omp "$csr" -c "$CSV" >/dev/null
            done
            ./build/pagerank_omp_float "$csr" -c "$CSV" >/dev/null
        fi
        if measures hybrid; then
            for s in $SHARES; do
                run_hybrid ./build/pagerank_hybrid double "$s" "$rep"
                run_hybrid ./build/pagerank_hybrid_float float "$s" "$rep"
            done
        fi
        echo "   repetition $rep done"
    done

    # The ranks themselves only need computing once.
    if measures omp; then
        ./build/pagerank_omp "$csr" -i "$ids" -o "$OUT/$graph.ranks.txt" >/dev/null
    fi
done

if measures hybrid && [ -n "$NSYS" ]; then
    "$PYTHON" tools/nsys_phases.py "$OUT" --nsys "$NSYS"
fi

echo
echo "wrote $CSV ($(( $(wc -l < "$CSV") - 1 )) runs)"
ls -1 "$OUT"/*.ranks.txt 2>/dev/null | while read -r f; do
    echo "wrote $f ($(du -h "$f" | cut -f1))"
done
