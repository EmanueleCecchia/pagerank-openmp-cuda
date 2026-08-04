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

To download:

```bash
mkdir -p data/snap && cd data/snap
curl -O https://snap.stanford.edu/data/wiki-Vote.txt.gz && gunzip wiki-Vote.txt.gz
curl -O https://snap.stanford.edu/data/web-Google.txt.gz && gunzip web-Google.txt.gz
```

## Project structure

- `data/` — downloaded datasets (gitignored)