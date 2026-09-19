#!/usr/bin/env python3
"""Show experimentally that the gather cost comes from the node numbering.

Relabels the nodes of a .csr graph at random and measures again.  The
permutation leaves the node count, the edge count and every single row length
untouched, and therefore the whole of Table tab:bins as well: the only thing
it destroys is the locality of the indices read from contrib.  If the times
go up, the gap between web-Google and web-BerkStan is a matter of numbering
and not of row shape.

Usage: python3 tools/locality_experiment.py data/snap/web-BerkStan.csr
       python3 tools/locality_experiment.py data/snap/web-BerkStan.csr -p 1 4 8 -r 5
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"PRCSR001"
SEME = 12345


def carica(path):
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: non e' un file .csr")
        n, m = (int(x) for x in np.fromfile(fh, dtype=np.uint64, count=2))
        row_ptr = np.fromfile(fh, dtype=np.uint64, count=n + 1)
        col_idx = np.fromfile(fh, dtype=np.uint32, count=m)
        out_deg = np.fromfile(fh, dtype=np.uint32, count=n)
    return n, m, row_ptr, col_idx, out_deg


def salva(path, row_ptr, col_idx, out_deg):
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        np.array([row_ptr.size - 1, col_idx.size], dtype=np.uint64).tofile(fh)
        row_ptr.astype(np.uint64).tofile(fh)
        col_idx.astype(np.uint32).tofile(fh)
        out_deg.astype(np.uint32).tofile(fh)


def permuta(n, m, row_ptr, col_idx, out_deg):
    """Relabel the nodes at random, keeping each row sorted afterwards."""
    rng = np.random.default_rng(SEME)
    nuova = rng.permutation(n).astype(np.uint32)      # nuova[v] = new label of v
    ordine = np.argsort(nuova)                        # ordine[k] = old node now labelled k
    grado = np.diff(row_ptr).astype(np.int64)

    grado_p = grado[ordine]
    row_ptr_p = np.zeros(n + 1, dtype=np.uint64)
    row_ptr_p[1:] = np.cumsum(grado_p)

    inizio = row_ptr[:-1].astype(np.int64)[ordine]
    presa = np.repeat(inizio - row_ptr_p[:-1].astype(np.int64), grado_p) + np.arange(m)
    col_idx_p = nuova[col_idx[presa]]

    # col_idx must stay ascending inside every row, the way the converter writes it
    riga = np.repeat(np.arange(n, dtype=np.int64), grado_p)
    col_idx_p = col_idx_p[np.lexsort((col_idx_p, riga))]

    assert np.array_equal(np.sort(grado_p), np.sort(grado)), "lunghezze alterate"
    return row_ptr_p, col_idx_p, out_deg[ordine]


def misura(binario, grafo, thread, ripetizioni, csv_tmp):
    """Minimum over several runs: system activity can only ever slow things down."""
    best = None
    for _ in range(ripetizioni):
        if csv_tmp.exists():
            csv_tmp.unlink()
        res = subprocess.run([str(binario), str(grafo), "-c", str(csv_tmp)],
                             capture_output=True, text=True,
                             env={**os.environ, "OMP_NUM_THREADS": str(thread)})
        if res.returncode != 0:
            sys.exit(f"{binario} e' uscito con {res.returncode}:\n{res.stderr}")
        with open(csv_tmp) as fh:
            r = list(csv.DictReader(fh))[-1]
        t = float(r["seconds_total"])
        if best is None or t < best[0]:
            best = (t, int(r["iterations"]))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grafo", type=Path)
    ap.add_argument("-b", "--binario", type=Path, default=None,
                    help="default: build/pagerank_omp (or .exe on Windows)")
    ap.add_argument("-p", "--thread", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("-r", "--ripetizioni", type=int, default=3)
    args = ap.parse_args()

    binario = args.binario
    if binario is None:
        for c in (Path("build/pagerank_omp"), Path("build/pagerank_omp.exe")):
            if c.exists():
                binario = c
                break
        else:
            sys.exit("binario non trovato: compila con make, o passa -b")

    n, m, row_ptr, col_idx, out_deg = carica(args.grafo)
    print(f"{args.grafo.stem}: N={n:,}  M={m:,}  binario={binario}")
    print("permuto le etichette dei nodi (N, M e ogni lunghezza di riga restano identici)")

    tmp = Path(tempfile.mkdtemp(prefix="localita-"))
    try:
        mescolato = tmp / (args.grafo.stem + "-mescolato.csr")
        salva(mescolato, *permuta(n, m, row_ptr, col_idx, out_deg))
        csv_tmp = tmp / "run.csv"

        print(f"\n{'thread':>7}{'originale':>12}{'mescolato':>12}{'divario':>10}"
              f"{'ns/arco orig.':>15}{'ns/arco mesc.':>15}")
        for p in args.thread:
            t0, it0 = misura(binario, args.grafo, p, args.ripetizioni, csv_tmp)
            t1, it1 = misura(binario, mescolato, p, args.ripetizioni, csv_tmp)
            if it0 != it1:
                print(f"  attenzione: iterazioni diverse ({it0} contro {it1})")
            print(f"{p:>7}{t0:>11.3f}s{t1:>11.3f}s{(t1/t0 - 1)*100:>9.0f}%"
                  f"{t0/it0/m*1e9:>15.2f}{t1/it1/m*1e9:>15.2f}")
        print(f"\nminimo su {args.ripetizioni} esecuzioni per configurazione")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
