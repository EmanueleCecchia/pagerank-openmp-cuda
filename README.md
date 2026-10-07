# PageRank on Hybrid Architectures (OpenMP + CUDA)

PageRank over large sparse graphs, in three versions built from one set of
sources: a sequential baseline, a pure-OpenMP one, and a hybrid OpenMP+CUDA
one, which with no work for the CPU is the GPU-only version.

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
| hybrid version | CUDA Toolkit (`nvcc`) |
| benchmark sweep | Linux, `bash`, `nproc`, `lscpu`; optionally Nsight Systems (`nsys`), to profile the hybrid runs |

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
hybrid ones only when `nvcc` is found, otherwise `make` builds the CPU ones alone:

| Executable | Compiled with | Purpose |
|---|---|---|
| `pagerank_seq` | — | sequential baseline (OpenMP pragmas ignored) |
| `pagerank_omp` | `-fopenmp` | parallel, double precision |
| `pagerank_omp_float` | `-fopenmp -DPAGERANK_FLOAT` | parallel, single precision |
| `pagerank_hybrid` | `nvcc`, `-fopenmp -DPAGERANK_CUDA` | GPU and CPU together, double precision |
| `pagerank_hybrid_float` | `nvcc`, `-fopenmp -DPAGERANK_CUDA -DPAGERANK_FLOAT` | GPU and CPU together, single precision |
| `csr_info` | — | statistics of a converted graph |

The GPU code is compiled for the GPU of the machine running `make`. To build
for another one, name its architecture, e.g. for an RTX 2080 Ti:

```bash
make CUDA_ARCH=-arch=sm_75
```

`make cuda` builds the hybrid executables alone, and fails saying why when
`nvcc` is missing. `make clean` removes `build/`.

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
| `-b T,W` | hybrid builds only: on the GPU, rows of up to T in-neighbours get a thread each, up to W a warp, longer ones a block | 16,256 |
| `-s VAL` | hybrid builds only: share of the edges for the CPU, which takes the longest rows first; 0 leaves every row to the GPU | 0.5 |

In the OpenMP and hybrid builds the thread count comes from `OMP_NUM_THREADS`.
Left unset, the run uses every available logical thread; set it to pick a specific number:

```bash
OMP_NUM_THREADS=4 ./build/pagerank_omp data/snap/web-Google.csr
```

The hybrid builds run on the first visible GPU and print which one; on a
machine with more than one, `CUDA_VISIBLE_DEVICES` picks it:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/pagerank_hybrid data/snap/web-Google.csr
```

In the hybrid builds the CPU gathers the longest rows, from the longest down
until they hold the share of the edges `-s` asks for, and the GPU the others;
the output says which rows the CPU took. `-s 0` gives the CPU
nothing, and is the GPU-only version: the same code, with nothing to copy
and no CPU thread started.

```bash
./build/pagerank_hybrid data/snap/web-Google.csr -s 0
```

A `-b` threshold T beyond the longest row gives every GPU row its own thread,
which is the naive kernel the classes are compared against:
`-s 0 -b 1000000,1000000`.

### Saving the results

`-o` writes every rank, one node per line, after a header recording how
the run was produced (build, precision, threads, iterations, timing, rank
sum; the hybrid builds add the device, the `-b` classes and the rows the CPU
took). Ranks are written in node order rather than sorted by rank, so that two
files line up line-by-line and can be diffed directly; values carry enough
digits to round-trip exactly. To view them by rank instead:

```bash
grep -v '^#' ranks.txt | sort -k2 -g -r | head
```

`-c` appends one row per run to a CSV — graph, nodes, edges, build,
precision, threads, damping, tolerance, iterations, converged, seconds total,
seconds per iteration, rank sum, share of the edges asked of the CPU (the
`-s` of the hybrid builds, 1 for the CPU ones; the CPU takes whole row
lengths at a time, so the share it actually gets can be smaller, and the
standard output and the `-o` header report it), classes (the `-b` of the
hybrid builds, as `16/256`; empty for the CPU ones) — writing the header only
when the file is created, so a sweep builds its own results table:

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
python3 tools/verify_pagerank.py data/snap/wiki-Vote.txt --c-executable build/pagerank_hybrid
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
the sequential, OpenMP and hybrid builds, each also in single precision but
the sequential, three repetitions each, and writes:

- `results/<machine>/bench.csv` — one row per run;
- `results/<machine>/<graph>.ranks.txt` — the rank vectors (gitignored).

The OpenMP thread counts follow the CPU: the powers of two up to the logical
CPUs, plus the physical cores and the logical CPUs themselves (ex. 1/2/4/8 on the
4-core/8-thread machine). The hybrid build runs on every logical CPU, once for
each share of the edges for the CPU: 0 (the GPU alone), 0.05, 0.1, 0.25, 0.5, 0.75
and 1 (every row with in-neighbours on the CPU; not the same as the OpenMP
build, since the GPU still computes `contrib` and the vectors still cross
the bus at every iteration).
Settings can be overridden from the environment — `GRAPHS`, `BUILDS`,
`THREADS`, `SHARES`, `REPS`, `DATA`, `OUT`:

```bash
MACHINE=machine2 GRAPHS="wiki-Vote web-Google" REPS=1 tools/run_benchmarks.sh
```

`BUILDS` picks which builds to run: `seq`, `omp`, `hybrid`. A sweep only
replaces the rows of `bench.csv` it measures again:

```bash
MACHINE=machine1 BUILDS=hybrid tools/run_benchmarks.sh
MACHINE=machine1 BUILDS=hybrid SHARES=1 tools/run_benchmarks.sh
```

`MACHINE` may name a sub-folder too, which keeps apart the GPUs of a machine
that has more than one: the CPU builds are measured once, in the machine's
folder, and the hybrid one once per GPU, picked with `CUDA_VISIBLE_DEVICES`,
in a folder of its own. That is how `results/` is laid out for Machine 3,
whose two RTX 2080 Ti sit on a PCIe x16 and an x8 link:

```bash
MACHINE=machine3 BUILDS="seq omp" tools/run_benchmarks.sh
CUDA_VISIBLE_DEVICES=0 MACHINE=machine3/x16 BUILDS=hybrid tools/run_benchmarks.sh
CUDA_VISIBLE_DEVICES=1 MACHINE=machine3/x8  BUILDS=hybrid tools/run_benchmarks.sh
```

and for Machine 2, one folder per GPU (`gtx1080ti`, `rtx2080ti`), with
hybrid runs only.

When `nsys` is installed, every run of the hybrid build goes under it,
tracing the CUDA calls only, and its report is kept in
`results/<machine>/nsys/` (gitignored; to look at one, open it with
`nsys-ui`, the graphical viewer of Nsight Systems). At the end
`tools/nsys_phases.py` reads from each report how that
run's iterations split up — the GPU's gather, the copy of `contrib` to the
host, the CPU's gather, the ranks going back, the waits — and writes
`results/<machine>/nsys_phases.csv`, one row for every hybrid row of
`bench.csv`, from the very same run, with its time copied for checking.
The profiler costs about 1% of the time. `NSYS=` runs without it.

The analysis tools read what the sweep produced:

```bash
python3 tools/plot_results.py
python3 tools/hybrid_table.py results/machine1
python3 tools/nsys_phases.py results/machine1
python3 tools/locality_stats.py data/snap/*.csr
```

`plot_results.py` turns the `results/<machine>/bench.csv` files into two figures,
speed-up and parallel efficiency, in `relazione/figure/`, with one curve per
machine in each. Without `--machine` it takes the folders right under
`results/` that have a `bench.csv`, which is where the CPU builds are.
To pick or rename them, give one `--machine` per machine, with its folder or CSV:
`--machine "Laptop 4c/8t" results/machine1 --machine "Workstation 12c/24t" results/machine3`.
`hybrid_table.py` prints the times of the hybrid build, one row per graph and
one column per share of the edges for the CPU, next to the best OpenMP time
in the same `bench.csv`: in a GPU's own sub-folder there is none, and that
column shows `-`.
`nsys_phases.py` rebuilds `nsys_phases.csv` from the reports, without running
anything.
`locality_stats.py` reports, per graph, the median index gap inside a row and
the cache lines the gather touches: their count, per edge and per line, and
the MiB one iteration asks for.

## Project structure

- `src/` — C and CUDA sources: `csr.*` (loader and validation), `pagerank.*`
  (the timed kernel), `pagerank_cuda.cu` (the same kernel on the GPU, with
  the CPU taking the longest rows),
  `main.c` (driver), `csr_info.c` (graph statistics)
- `tools/` — Python and shell helpers: conversion, verification, benchmark
  sweep, figures, table of the hybrid times, phases of the hybrid iterations,
  locality statistics
- `relazione/` — the report, LaTeX source and compiled PDF
- `results/` — one folder per machine, and inside it one per GPU when the
  machine has more than one: `bench.csv` (every run), `nsys_phases.csv` (the
  phases of the hybrid runs), and the gitignored rank vectors and nsys reports
- `data/` — the datasets (gitignored)
- `build/` — the executables (gitignored)
