# PageRank on Hybrid Architectures (OpenMP + CUDA)

PageRank over large sparse graphs, in three versions built from one set of
sources: a sequential baseline, a pure-OpenMP one, and a hybrid OpenMP+CUDA
one (in development).

This file is the usage guide: how to build, get the data, run, check the
results and reproduce the experiments. The problem, the design choices and
the measurements are in [`relazione/relazione.pdf`](relazione/relazione.pdf)
(in Italian).

## Requirements

| For | Needs |
|---|---|
| building | `gcc` with OpenMP support, `make` |
| dataset conversion | Python 3, `numpy` |
| correctness check | `networkx`, `scipy` |
| figures | `matplotlib` |
| hybrid version | CUDA Toolkit |

## Building

```bash
make
```

Produces four executables in `build/`, all from the same sources:

| Executable | Compiled with | Purpose |
|---|---|---|
| `pagerank_seq` | — | sequential baseline (OpenMP pragmas ignored) |
| `pagerank_omp` | `-fopenmp` | parallel, double precision |
| `pagerank_omp_float` | `-fopenmp -DPAGERANK_FLOAT` | parallel, single precision |
| `csr_info` | — | statistics of a converted graph |

`make clean` removes `build/`.

## Datasets

Directed edge lists from the [SNAP](https://snap.stanford.edu/data/)
collection, expected under `data/snap/` and not committed to the repo:
[wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html),
[web-Google](https://snap.stanford.edu/data/web-Google.html),
[web-BerkStan](https://snap.stanford.edu/data/web-BerkStan.html),
[soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html).

```bash
mkdir -p data/snap && cd data/snap
for g in wiki-Vote web-Google web-BerkStan soc-LiveJournal1; do
    curl -O https://snap.stanford.edu/data/$g.txt.gz && gunzip $g.txt.gz
done
cd ../..
```

### Converting to CSR

The executables read a binary CSR file, produced once per graph:

```bash
python3 tools/snap_to_csr.py data/snap/web-Google.txt
```

This writes `web-Google.csr` (the graph) and `web-Google.ids` (the original
SNAP id of each remapped node, used only for reporting). Options:
`-o path/graph.csr` to choose the output path, `--drop-self-loops` to discard
`u -> u` edges.

The converter holds the whole edge list in memory at roughly 100 bytes per
edge, so soc-LiveJournal1 peaks near 6.7 GB.

To look inside a converted graph — node and edge counts, dangling nodes,
degree extremes, row-length distribution:

```bash
./build/csr_info data/snap/web-Google.csr data/snap/web-Google.ids
```

## Running

```bash
./build/pagerank_omp data/snap/web-Google.csr -i data/snap/web-Google.ids
```

| Option | Meaning | Default |
|---|---|---|
| `-i path/graph.ids` | companion `.ids` file, to report original SNAP ids | — |
| `-d VAL` | damping factor | 0.85 |
| `-t VAL` | L1 convergence tolerance | 1e-6 |
| `-n NUM` | maximum iterations | 100 |
| `-k NUM` | how many top nodes to print | 10 |
| `-o path/ranks.txt` | write every rank to that file | — |
| `-c path/bench.csv` | append one CSV row of run details to that file | — |

The thread count comes from `OMP_NUM_THREADS`.
Left unset, the run uses every available logical thread; set it to pick a specific number:

```bash
OMP_NUM_THREADS=4 ./build/pagerank_omp data/snap/web-Google.csr
```

### Saving the results

`-o` writes every rank, one node per line, after a header recording how
the run was produced (build, precision, threads, iterations, timing, rank
sum). Ranks are written in node order rather than sorted by rank, so that two
files line up line-by-line and can be diffed directly; values carry enough
digits to round-trip exactly. To view them by rank instead:

```bash
grep -v '^#' ranks.txt | sort -k2 -g -r | head
```

`-c` appends one row per run to a CSV — graph, nodes, edges, build,
precision, threads, damping, tolerance, iterations, converged, seconds total,
seconds per iteration, rank sum — writing the header only when the file is
created, so a sweep builds its own results table:

```bash
for t in 1 2 4 8; do
    OMP_NUM_THREADS=$t ./build/pagerank_omp data/snap/web-Google.csr -c bench.csv
done
```

## Verifying correctness

```bash
python3 tools/verify_pagerank.py data/snap/wiki-Vote.txt
python3 tools/verify_pagerank.py data/snap/web-Google.txt --no-dense
```

Recomputes PageRank by routes sharing no code with the project and compares
them against the C executable; exits non-zero on any mismatch, so it works as
a regression test. `--no-dense` skips the dense N×N reference, which only fits
in memory for the smallest graph, and checks against networkx alone: that is
the form to use on the larger graphs.

| Option | Meaning | Default |
|---|---|---|
| `--c-executable PATH` | which build of the C code to check | `build/pagerank_seq` |
| `--csr path/graph.csr`, `--ids path/graph.ids` | companion files | edge list with the suffix replaced |
| `-d`, `-t`, `-n` | damping, L1 tolerance, maximum iterations | 0.85, 1e-12, 500 |
| `-k NUM` | how many top ranks to compare by order | 100 |
| `--value-tolerance VAL` | allowed difference per rank | 1e-8 |
| `--max-nodes NUM` | refuse the dense reference above this size | 15000 |

## Reproducing the experiments

```bash
tools/run_benchmarks.sh
```

Runs every graph against the sequential, OpenMP (1/2/4/8 threads) and float
builds, three repetitions each, and writes:

- `results/bench.csv` — one row per run;
- `results/<graph>.ranks.txt` — the rank vectors (gitignored).

Settings can be overridden from the environment — `GRAPHS`, `THREADS`,
`REPS`, `DATA`, `OUT`:

```bash
GRAPHS="wiki-Vote web-Google" REPS=1 tools/run_benchmarks.sh
```

The two analysis tools read what the sweep produced:

```bash
python3 tools/plot_results.py
python3 tools/locality_stats.py data/snap/*.csr
```

`plot_results.py` turns `results/bench.csv` into the speed-up and efficiency
figures, written to `relazione/figure/scalabilita.pdf` and `.png`.
`locality_stats.py` reports, per graph, the median index gap inside a row and
the cache lines the gather touches per edge.

## Project structure

- `src/` — C sources: `csr.*` (loader and validation), `pagerank.*` (the
  timed kernel), `main.c` (driver), `csr_info.c` (graph statistics)
- `tools/` — Python and shell helpers: conversion, verification, benchmark
  sweep, figures, locality statistics
- `relazione/` — the report, LaTeX source and compiled PDF
- `results/` — `bench.csv` with every run; the rank vectors are gitignored
- `data/` — the datasets (gitignored)
- `build/` — the executables (gitignored)
