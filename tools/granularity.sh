#!/usr/bin/env bash
# Kernel ingenuo (un thread per riga) contro le classi 16/256, GPU sola,
# doppia precisione. Una cartella per configurazione: nsys_phases.py ci
# lavora senza modifiche.
set -euo pipefail
: "${MACHINE:?imposta MACHINE, es. machine1 o machine3/x16}"
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
        echo "   $g fatto"
    done
    python3 tools/nsys_phases.py "$dir"
done