# PageRank on Hybrid Architectures (OpenMP + CUDA)

Implementation and profiling of the PageRank algorithm, benchmarking a
pure-OpenMP (CPU-only) implementation against a hybrid OpenMP+CUDA
implementation that splits work between host and device.

## Datasets

Real-world graphs from the [SNAP](https://snap.stanford.edu/data/) collection,
used in increasing order of size. Each is a plain-text directed edge list
(`FromNodeId<TAB>ToNodeId`) stored under `data/snap/` (gitignored — not
committed to the repo).

| Dataset | Nodes | Edges | Source |
|---|---|---|---|
| wiki-Vote | 7,115 | 103,689 | https://snap.stanford.edu/data/wiki-Vote.html |
| web-Google | 875,713 | 5,105,039 | https://snap.stanford.edu/data/web-Google.html |
| web-BerkStan | 685,230 | 7,600,595 | https://snap.stanford.edu/data/web-BerkStan.html |
| soc-LiveJournal1 | 4,847,571 | 68,993,773 | https://snap.stanford.edu/data/soc-LiveJournal1.html |

To download:

```bash
mkdir -p data/snap && cd data/snap
for g in wiki-Vote web-Google web-BerkStan soc-LiveJournal1; do
    curl -O https://snap.stanford.edu/data/$g.txt.gz && gunzip $g.txt.gz
done
```

## Preparing the data

PageRank pulls rank along *incoming* edges, so the graphs are converted once
into a binary CSR holding the transpose (row `v` = in-neighbours of `v`) plus
the out-degrees. Converting up front keeps text parsing out of the timed
region.

```bash
python3 tools/snap_to_csr.py data/snap/wiki-Vote.txt
```

This writes `wiki-Vote.csr` (the graph) and `wiki-Vote.ids` (the original SNAP
id of each remapped node, used only for reporting). Requires numpy. The
converter holds the whole edge list in memory at roughly 100 bytes per edge —
soc-LiveJournal1 peaks near 6.7 GB.

## Building

```bash
make
```

Produces four binaries in `build/`, all from the same sources:

| Binary | Build | Purpose |
|---|---|---|
| `pagerank_seq` | no `-fopenmp` | sequential baseline |
| `pagerank_omp` | `-fopenmp` | parallel, double precision |
| `pagerank_omp_float` | `-fopenmp -DPAGERANK_FLOAT` | parallel, single precision |
| `csr_info` | — | prints a converted graph's statistics |

`pagerank.c` carries the OpenMP pragmas; compiled without `-fopenmp` the
compiler ignores them and emits ordinary serial loops. The sequential baseline
is therefore the same code, not a separate implementation, and it avoids the
thread-management overhead that `OMP_NUM_THREADS=1` would still pay.

## Running

```bash
./build/pagerank_omp data/snap/web-Google.csr -i data/snap/web-Google.ids
```

Options: `-d` damping (default 0.85), `-t` L1 tolerance (default 1e-6),
`-n` maximum iterations (default 100), `-k` how many top nodes to print,
`-i` the `.ids` file. Thread count comes from `OMP_NUM_THREADS`.

### Saving results

`-o FILE` writes every rank, one node per line, after a header recording how
the run was produced (build, precision, threads, iterations, timing, rank sum):

```bash
./build/pagerank_omp data/snap/web-Google.csr -i data/snap/web-Google.ids -o ranks.txt
```

Ranks are written in node order, not sorted by rank, so that two files line up
line-by-line and can be compared directly — sorting would order near-ties
differently between implementations and make a diff meaningless. To view them
by rank instead: `grep -v '^#' ranks.txt | sort -k2 -g -r | head`.

Values carry enough digits to round-trip exactly, which is what makes it
possible to check one implementation against another across the whole vector
rather than just the top few. Comparing the OpenMP build against the
sequential one on web-Google gives a maximum relative difference of 2e-14
(floating-point summation order), and the float build 2e-7 (float's machine
epsilon).

`-c FILE` appends one CSV row per run instead, so a sweep builds its own
results table:

```bash
for t in 1 2 4 8; do
    OMP_NUM_THREADS=$t ./build/pagerank_omp data/snap/web-Google.csr -c bench.csv
done
```

## Reproducing the experiments

```bash
tools/run_benchmarks.sh
```

Runs every graph against the sequential, OpenMP (1/2/4/8 threads) and float
builds, three repetitions each, and writes:

- `results/bench.csv` — one row per run; take the minimum per configuration,
  since the fastest run is the one least disturbed by other activity
- `results/<graph>.ranks.txt` — the ranks themselves (gitignored: the
  soc-LiveJournal1 vector alone is ~150 MB)

Settings can be overridden from the environment, e.g.
`GRAPHS="wiki-Vote web-Google" REPS=1 tools/run_benchmarks.sh`.

## Verifying correctness

```bash
python3 tools/verify_pagerank.py data/snap/wiki-Vote.txt
```

Recomputes PageRank by routes sharing no code with the project, then compares
against the C binary. Four checks:

- the binary CSR really is the transpose of an independently parsed edge list;
- a **dense N×N** reference — no CSR, no gather loop, so a bug in the sparse
  representation cannot hide;
- **networkx**, a third-party implementation, against that dense reference:
  the dense code is independent in method but shares an author with the code
  it checks, so this rules out the same misreading of the algorithm appearing
  in both (skipped automatically if networkx or scipy is missing);
- the C binary itself — that it **converged** rather than hitting the
  iteration limit, that the top-k ranking and values match, and that the ranks
  sum to 1 (which fails if dangling-node mass is mishandled).

Exits non-zero on any mismatch, so it works as a regression test. The
convergence check matters: a run stopped at the iteration limit can still
produce the right *ordering* while its values are far from settled.

The networkx check needs `networkx` and `scipy` (networkx 3.x routes
`pagerank()` through scipy).

The dense matrix costs N² × 8 bytes, so this only works on the smallest graph:
wiki-Vote needs ~390 MiB, web-Google would need ~5.6 TiB. That is why
wiki-Vote is in the ladder — not for performance, but as the one graph where a
brute-force answer is computable at all.

Use `--binary build/pagerank_omp` to check the parallel build instead.

## Project structure

- `src/` — C sources (`csr.*` loader, `pagerank.*` kernel, `main.c` driver)
- `tools/` — Python helpers (dataset conversion, correctness verification)
- `data/` — downloaded datasets (gitignored)
- `build/` — compiled binaries (gitignored)
