#!/usr/bin/env bash
# The naive kernel (one thread per row) against the granularity classes
# 16/256, on the GPU alone (-s 0) and in double precision.  One folder per
# configuration under results/<machine>/granularity/, each with its own
# bench.csv and nsys reports, so that tools/nsys_phases.py reads it as is.
#
#   MACHINE=machine1 tools/granularity.sh
#   CUDA_VISIBLE_DEVICES=0 MACHINE=machine3/x16 tools/granularity.sh
set -euo pipefail
: "${MACHINE:?set MACHINE to name the results of this machine, e.g. machine1 or machine3/x16}"
GRAPHS="web-NotreDame web-Stanford web-Google web-BerkStan cit-Patents wiki-topcats soc-Pokec soc-LiveJournal1"
CONFIGS="naive:1000000,1000000 class-16-256:16,256"

for cfg in $CONFIGS; do
    name=${cfg%%:*}; b=${cfg#*:}
    dir=results/$MACHINE/granularity/$name
    mkdir -p "$dir/nsys"
    rm -f "$dir/bench.csv" "$dir"/nsys/*.nsys-rep
    echo "== $name (-b $b)"
    for g in $GRAPHS; do
        for r in 1 2 3; do
            nsys profile --trace=cuda --sample=none --cpuctxsw=none --show-output=false \
                --force-overwrite true -o "$dir/nsys/$g-double-s0-r$r" \
                ./build/pagerank_hybrid "data/snap/$g.csr" -s 0 -b "$b" -c "$dir/bench.csv" >/dev/null
        done
        echo "   $g done"
    done
    python3 tools/nsys_phases.py "$dir"
done