# PageRank on Hybrid Architectures (OpenMP + CUDA)

PageRank over large sparse graphs, in four versions built from one set of
sources: a sequential baseline, a pure-OpenMP one, a GPU-only CUDA one, and a
hybrid OpenMP+CUDA one (in development).

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
| GPU versions | CUDA Toolkit (`nvcc`) |

To create the conda environment:

```bash
conda env create -f environment.yml
conda activate pagerank
```

## Building

```bash
make
```

Produces these executables in `build/`, all from the same sources; the two
GPU ones only when `nvcc` is found, otherwise `make` builds the CPU ones alone:

| Executable | Compiled with | Purpose |
|---|---|---|
| `pagerank_seq` | — | sequential baseline (OpenMP pragmas ignored) |
| `pagerank_omp` | `-fopenmp` | parallel, double precision |
| `pagerank_omp_float` | `-fopenmp -DPAGERANK_FLOAT` | parallel, single precision |
| `pagerank_cuda` | `nvcc`, `-DPAGERANK_CUDA` | GPU only, double precision |
| `pagerank_cuda_float` | `nvcc`, `-DPAGERANK_CUDA -DPAGERANK_FLOAT` | GPU only, single precision |
| `csr_info` | — | statistics of a converted graph |

The GPU code is compiled for the GPU of the machine running `make`. To build
for another one, name its architecture, e.g. for an RTX 2080 Ti:

```bash
make CUDA_ARCH=-arch=sm_75
```

`make cuda` builds the GPU executables alone, and fails saying why when `nvcc`
is missing. `make clean` removes `build/`.

## Datasets

Directed edge lists from the [SNAP](https://snap.stanford.edu/data/)
collection, expected under `data/snap/` and not committed to the repo:
[wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html),
[web-NotreDame](https://snap.stanford.edu/data/web-NotreDame.html),
[web-Stanford](https://snap.stanford.edu/data/web-Stanford.html),
[web-Google](https://snap.stanford.edu/data/web-Google.html),
[web-BerkStan](https://snap.stanford.edu/data/web-BerkStan.html),
[cit-Patents](https://snap.stanford.edu/data/cit-Patents.html),
[wiki-topcats](https://snap.stanford.edu/data/wiki-topcats.html),
[soc-Pokec](https://snap.stanford.edu/data/soc-Pokec.html),
[soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html).

```bash
mkdir -p data/snap && cd data/snap
for g in wiki-Vote web-NotreDame web-Stanford web-Google web-BerkStan \
         cit-Patents wiki-topcats soc-LiveJournal1; do
    curl -O https://snap.stanford.edu/data/$g.txt.gz && gunzip $g.txt.gz
done
# soc-Pokec is published under another file name
curl -o soc-Pokec.txt.gz https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
gunzip soc-Pokec.txt.gz
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

To convert every downloaded graph at once:

```bash
for f in data/snap/*.txt; do python3 tools/snap_to_csr.py "$f"; done
```

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
| `-b T,W` | GPU builds only: rows of up to T in-neighbours get a thread each, up to W a warp, longer ones a block | 16,256 |

In the OpenMP builds the thread count comes from `OMP_NUM_THREADS`.
Left unset, the run uses every available logical thread; set it to pick a specific number:

```bash
OMP_NUM_THREADS=4 ./build/pagerank_omp data/snap/web-Google.csr
```

The GPU builds run on the first visible GPU and print which one; on a machine
with more than one, `CUDA_VISIBLE_DEVICES` picks it:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/pagerank_cuda data/snap/web-Google.csr
```

A `-b` threshold T beyond the longest row gives every row its own thread,
which is the naive kernel the classes are compared against:
`-b 1000000,1000000`.

### Saving the results

`-o` writes every rank, one node per line, after a header recording how
the run was produced (build, precision, threads, iterations, timing, rank
sum; the GPU builds add the device and the `-b` classes). Ranks are written in node order rather than sorted by rank, so that two
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
the form to use on the larger graphs. To check another build, name it:

```bash
python3 tools/verify_pagerank.py data/snap/wiki-Vote.txt --c-executable build/pagerank_cuda
```

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
MACHINE=machine2 tools/run_benchmarks.sh
```

Every machine the experiments run on gets its own folder, named by `MACHINE`.
The sweep runs every graph against
the sequential, OpenMP and float builds, three repetitions each, and writes:

- `results/<machine>/bench.csv` — one row per run;
- `results/<machine>/<graph>.ranks.txt` — the rank vectors (gitignored).

The OpenMP thread counts follow the CPU: the powers of two up to the logical
CPUs, plus the physical cores and the logical CPUs themselves (ex. 1/2/4/8 on the
4-core/8-thread machine).
Settings can be overridden from the environment — `GRAPHS`, `THREADS`,
`REPS`, `DATA`, `OUT`:

```bash
MACHINE=machine2 GRAPHS="wiki-Vote web-Google" REPS=1 tools/run_benchmarks.sh
```

The two analysis tools read what the sweep produced:

```bash
python3 tools/plot_results.py
python3 tools/locality_stats.py data/snap/*.csr
```

`plot_results.py` turns the `results/<machine>/bench.csv` files into two figures,
speed-up and parallel efficiency, in `relazione/figure/`, with one curve per
machine in each.
To pick or rename them, give one `--machine` per machine, with its folder or CSV:
`--machine "Laptop 4c/8t" results/machine1 --machine "Workstation 12c/24t" results/machine3`.
`locality_stats.py` reports, per graph, the median index gap inside a row and
the cache lines the gather touches: their count, per edge and per line, and
the MiB one iteration asks for.

## Project structure

- `src/` — C and CUDA sources: `csr.*` (loader and validation), `pagerank.*`
  (the timed kernel), `pagerank_cuda.cu` (the same kernel on the GPU),
  `main.c` (driver), `csr_info.c` (graph statistics)
- `tools/` — Python and shell helpers: conversion, verification, benchmark
  sweep, figures, locality statistics
- `relazione/` — the report, LaTeX source and compiled PDF
- `results/` — one folder per machine, with `bench.csv` (every run) and the
  gitignored rank vectors;
- `data/` — the datasets (gitignored)
- `build/` — the executables (gitignored)
