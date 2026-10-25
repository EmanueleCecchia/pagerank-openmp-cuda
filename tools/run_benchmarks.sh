#!/usr/bin/env bash
# Runs the experiment sweep on this machine and writes, under results/<machine>/:
#
#   bench.csv           one row per run (graph, build, threads, timing)
#   <graph>.ranks.txt   the ranks, from the OpenMP build (gitignored)
#   nsys/               with nsys installed, a report per hybrid run (gitignored),
#   nsys_phases.csv     and the phases of its iterations (tools/nsys_phases.py)
#
# MACHINE is required, so a sweep never overwrites another machine's results;
# it may name a sub-folder per GPU, picked with CUDA_VISIBLE_DEVICES:
#   MACHINE=machine1 tools/run_benchmarks.sh
#   CUDA_VISIBLE_DEVICES=0 MACHINE=machine3/x16 BUILDS=hybrid tools/run_benchmarks.sh
#
# Each configuration runs REPS times, and the tables take the minimum, the run
# least disturbed by other activity.  A sweep replaces in bench.csv only the
# rows it measures again (graph, build and, for the hybrid one, share).  Other
# settings come from the environment: GRAPHS, BUILDS (seq, omp, hybrid),
# THREADS, SHARES, REPS, DATA, OUT, NSYS (empty: no nsys), PYTHON.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -z "${MACHINE:-}" ]; then
    echo "set MACHINE to name this machine's results, e.g." >&2
    echo "  MACHINE=machine2 tools/run_benchmarks.sh" >&2
    echo "already measured: $(cd results 2>/dev/null && ls -d */ 2>/dev/null | tr -d / | xargs)" >&2
    exit 1
fi

# Powers of two up to the logical CPUs, plus the physical cores and the logical
# CPUs: 1 2 4 8 on 4 cores/8 threads, 1 2 4 8 12 16 24 on 12 cores/24 threads.
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
# Shares of the edges for the CPU in the hybrid build: 0 is the GPU alone, 1
# every row with in-neighbours on the CPU.
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

# Drop the rows about to be measured again: graph (last part of the .csr path),
# build and, for the hybrid one, share (column 14, compared as a number).
# LC_ALL=C: with a decimal comma mawk reads 0.25 as 0 and would drop too much.
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
# The same for their nsys reports.
if measures hybrid && [ -n "$NSYS" ]; then
    mkdir -p "$NSYS_DIR"
    for graph in $GRAPHS; do
        for s in $SHARES; do
            rm -f "$NSYS_DIR/$graph"-double-s"$s"-r*.nsys-rep "$NSYS_DIR/$graph"-float-s"$s"-r*.nsys-rep
        done
    done
fi

# One hybrid run on every logical CPU, under nsys if any, tracing the CUDA
# calls only: sampling the CPU would slow down the CPU's share.
# --show-output=false: otherwise nsys 2022-2023 starts nsys-tee, which takes a
# CPU away from the OpenMP threads (about 0.1 s per run on Machine 2).
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
            # no OpenMP at all, not even one thread
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
