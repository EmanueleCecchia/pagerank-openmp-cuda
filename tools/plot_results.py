#!/usr/bin/env python3
"""Genera le figure di scalabilità della relazione da results/bench.csv.

Produce relazione/figure/scalabilita.pdf (vettoriale, per LaTeX) e un PNG
di controllo. Il riferimento per lo speed-up è la build sequenziale vera
(compilata senza -fopenmp), non la build OpenMP limitata a un thread.

wiki-Vote e' escluso: con tempi dell'ordine del millesimo di secondo la
dispersione fra ripetizioni raggiunge un fattore 51, quindi misurerebbe
l'avvio dei thread e non l'algoritmo.
"""

import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palette categorica in ordine fisso (slot 1-3), validata: separazione CVD
# min 10.1 dE, separazione a vista normale min 27.6 dE.
SERIE = [
    ("web-Google",       "#2a78d6"),
    ("web-BerkStan",     "#eb6834"),
    ("soc-LiveJournal1", "#1baf7a"),
]
THREADS = [1, 2, 4, 8]
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#d4d4d4"


def carica(path):
    """Minimo per configurazione: l'attivita' di sistema puo' solo rallentare."""
    best = collections.defaultdict(lambda: float("inf"))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            g = r["graph"].split("/")[-1].replace(".csr", "")
            k = (g, r["build"], r["precision"], int(r["threads"]))
            best[k] = min(best[k], float(r["seconds_total"]))
    return best


def main():
    radice = Path(__file__).resolve().parent.parent
    best = carica(radice / "results" / "bench.csv")
    uscita = radice / "relazione" / "figure"
    uscita.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.set_xticklabels(THREADS)
        ax.set_xlabel("Thread OpenMP")
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)

    # riferimento ideale: linea sottile, tratteggiata, chiaramente non un dato
    ax1.plot(THREADS, THREADS, "--", color=INK2, linewidth=1.0,
             dashes=(4, 3), zorder=1)
    ax1.annotate("ideale (lineare)", xy=(3.0, 3.0), xytext=(0, 5),
                 textcoords="offset points", color=INK2, fontsize=7.2,
                 rotation=38, ha="center")
    ax2.axhline(1.0, ls="--", color=INK2, linewidth=1.0, dashes=(4, 3), zorder=1)

    for nome, colore in SERIE:
        seq = best[(nome, "seq", "double", 1)]
        sp = [seq / best[(nome, "omp", "double", p)] for p in THREADS]
        ef = [s / p for s, p in zip(sp, THREADS)]

        for ax, y in ((ax1, sp), (ax2, ef)):
            ax.plot(THREADS, y, "-o", color=colore, linewidth=2.0,
                    markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.9, zorder=3, label=nome)

    ax1.set_ylabel("Speed-up rispetto al sequenziale")
    ax1.set_ylim(0, 4.6)
    ax1.set_title("(a) Speed-up", fontsize=9.5, pad=6)

    ax2.set_ylabel("Efficienza parallela")
    ax2.set_ylim(0, 1.12)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax2.set_title("(b) Efficienza", fontsize=9.5, pad=6)

    maniglie, etichette = ax1.get_legend_handles_labels()
    fig.legend(maniglie, etichette, loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for est in ("pdf", "png"):
        fig.savefig(uscita / f"scalabilita.{est}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"scritto {uscita/'scalabilita.pdf'} e .png")

    for nome, _ in SERIE:
        seq = best[(nome, "seq", "double", 1)]
        sp = [seq / best[(nome, "omp", "double", p)] for p in THREADS]
        print(f"  {nome:<18} speed-up " + " ".join(f"{p}t={s:.2f}x" for p, s in zip(THREADS, sp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
