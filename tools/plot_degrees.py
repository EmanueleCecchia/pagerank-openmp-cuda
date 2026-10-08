#!/usr/bin/env python3
"""Genera la figura e la tabella delle classi di righe della GPU dai file .csr.

Produce in relazione/figure/ predecessori.pdf, con un PNG di controllo: per
ogni grafo, su assi log-log, quanti nodi hanno esattamente k predecessori,
cioe' una riga lunga k nel CSR trasposto, un punto per ogni k presente.  Il
punto ha il colore della classe in cui la GPU mette quelle righe: un thread,
un warp o un blocco per riga, con le soglie di -b (16 e 256 per default),
segnate da due linee punteggiate.  Nella coda molti k hanno un solo nodo, e
i loro punti si allineano sull'1: sono gli hub, uno per uno.  I nodi senza
predecessori non compaiono: su un asse logaritmico lo 0 non ha posto.

Stampa inoltre, per ogni grafo, la quota di righe e di archi di ciascuna
classe, a quota 0 (nessuna riga alla CPU): i numeri della tabella delle
classi nella relazione.  Le righe vuote stanno nella classe dei thread,
come in pagerank_cuda.cu.

    python3 tools/plot_degrees.py
    python3 tools/plot_degrees.py --soglie 32,512 --data data/snap

Un pannello per grafo, come nelle figure di plot_results.py, di cui riprende
grafi, colori e stile.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from plot_results import GRAFI, GRID, INK, INK2, STILI, plt, salva

MAGIC = b"PRCSR001"
CLASSI = ("thread", "warp", "blocco")


def lunghezze(path):
    """Lunghezza di ogni riga del .csr (vedi tools/snap_to_csr.py per il
    formato).  Legge solo row_ptr: col_idx non serve, e sul grafo piu' grande
    pesa 263 MiB."""
    with open(path, "rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            raise SystemExit(f"{path}: non e' un file .csr")
        n, _ = (int(x) for x in np.fromfile(fh, dtype=np.uint64, count=2))
        row_ptr = np.fromfile(fh, dtype=np.uint64, count=n + 1)
    return np.diff(row_ptr).astype(np.int64)


def classe(lun, t, w):
    """0 thread, 1 warp, 2 blocco, come row_class() in pagerank_cuda.cu."""
    return np.where(lun <= t, 0, np.where(lun <= w, 1, 2))


def tabella(dati, t, w):
    print(f"\nclassi con soglie {t} e {w}, a quota 0: % righe / % archi")
    print(f"{'grafo':<18}" + "".join(f"{c:>16}" for c in CLASSI))
    for nome, lun in dati:
        c = classe(lun, t, w)
        m = lun.sum()
        celle = "".join(f"{100 * np.mean(c == k):>8.1f}{100 * lun[c == k].sum() / m:>8.1f}"
                        for k in range(3))
        print(f"{nome:<18}{celle}")


def figura(dati, t, w, uscita):
    fig, assi = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True,
                             layout="constrained")
    colori = np.array([STILI[c][0] for c in range(3)])
    for ax, (nome, lun) in zip(assi.flat, dati):
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)
        ax.set_title(nome, fontsize=8.5, pad=4)
        # un punto per ogni numero di predecessori presente nel grafo
        k, nodi = np.unique(lun[lun > 0], return_counts=True)
        ax.scatter(k, nodi, s=2.5, c=colori[classe(k, t, w)], linewidths=0,
                   zorder=3)
        # le soglie fra le classi, a meta' fra l'ultima riga di una e la prima
        # dell'altra
        for s in (t, w):
            ax.axvline(s + 0.5, color=INK2, linewidth=0.8, linestyle=(0, (1, 2)),
                       zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.7, 4e5)
        ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5])
        ax.set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
        ax.set_ylim(0.6, 2e6)
        ax.set_yticks([1, 1e2, 1e4, 1e6])
        ax.set_yticklabels(["1", "100", "10⁴", "10⁶"])
        ax.minorticks_off()

    voci = [plt.Line2D([], [], color=c, marker="o", linestyle="none", markersize=5)
            for c in colori]
    nomi = [f"un thread per riga (fino a {t})", f"un warp (fino a {w})",
            f"un blocco (oltre {w})"]
    fig.legend(voci, nomi, loc="outside upper center", ncols=3, frameon=False,
               fontsize=8)
    fig.supxlabel("Predecessori", fontsize=8.5)
    fig.supylabel("Nodi con quel numero di predecessori", fontsize=8.5)
    salva(fig, uscita, "predecessori")
    plt.close(fig)


def main(argv=None):
    radice = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--soglie", default="16,256", metavar="T,W",
                        help="le soglie di -b della versione ibrida (default 16,256)")
    parser.add_argument("--data", type=Path, default=radice / "data" / "snap",
                        help="cartella dei file .csr (default data/snap)")
    args = parser.parse_args(argv)
    t, w = (int(x) for x in args.soglie.split(","))

    tutti = ["wiki-Vote"] + GRAFI
    mancano = [g for g in tutti if not (args.data / f"{g}.csr").is_file()]
    if mancano:
        parser.error(f"mancano in {args.data}: {', '.join(mancano)}")
    dati = [(g, lunghezze(args.data / f"{g}.csr")) for g in tutti]

    tabella(dati, t, w)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
    })
    uscita = radice / "relazione" / "figure"
    uscita.mkdir(parents=True, exist_ok=True)
    figura(dati[1:], t, w, uscita)   # wiki-Vote fuori, come nelle altre figure
    return 0


if __name__ == "__main__":
    sys.exit(main())
