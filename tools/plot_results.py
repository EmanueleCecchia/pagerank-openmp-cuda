#!/usr/bin/env python3
"""Genera la figura di scalabilità della relazione da results/bench.csv.

Produce relazione/figure/scalabilita.pdf (vettoriale, per LaTeX) e un PNG
di controllo. Il riferimento per lo speed-up è la build sequenziale vera
(compilata senza -fopenmp), non la build OpenMP limitata a un thread.

Un pannello per grafo, con gli assi in comune: otto curve sovrapposte in un
solo grafico si confonderebbero proprio fra 2 e 4 thread, dove si decide la
lettura.  wiki-Vote e' escluso: i suoi tempi sono sotto il millisecondo.
"""

import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Prima riga i quattro grafi web, seconda le quattro istanze grandi.
GRAFI = [
    "web-NotreDame", "web-Stanford", "web-Google", "web-BerkStan",
    "cit-Patents", "wiki-topcats", "soc-Pokec", "soc-LiveJournal1",
]
THREADS = [1, 2, 4, 8]
# Una sola serie per pannello: basta un colore, il titolo dice di che grafo si tratta.
SERIE = "#2a78d6"
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


def speedup(best, nome):
    seq = best[(nome, "seq", "double", 1)]
    return [seq / best[(nome, "omp", "double", p)] for p in THREADS]


def virgola(x):
    return f"{x:.2f}".replace(".", ",")


def main():
    radice = Path(__file__).resolve().parent.parent
    best = carica(radice / "results" / "bench.csv")
    uscita = radice / "relazione" / "figure"
    uscita.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
    })

    fig, assi = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True,
                            layout="constrained")

    for ax, nome in zip(assi.flat, GRAFI):
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(THREADS)
        ax.set_xticklabels(THREADS)
        ax.set_ylim(0, 4.6)
        ax.set_yticks([0, 1, 2, 3, 4])
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)

        # riferimento ideale: linea sottile, tratteggiata, chiaramente non un dato
        ax.plot(THREADS, THREADS, "--", color=INK2, linewidth=0.9,
                dashes=(4, 3), zorder=1)

        sp = speedup(best, nome)
        ax.plot(THREADS, sp, "-o", color=SERIE, linewidth=1.8,
                markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.8, zorder=3)

        # un'etichetta sola per pannello: il massimo, che e' il dato riportato nel testo.
        # Dal lato dove la curva non passa: sull'ultimo punto sopra e allineata a
        # destra, per non uscire dal pannello; altrove sotto, lontano dalla linea ideale.
        i = max(range(len(sp)), key=sp.__getitem__)
        ultimo = i == len(THREADS) - 1
        ax.annotate(f"{virgola(sp[i])}×", xy=(THREADS[i], sp[i]),
                    xytext=(4, 6) if ultimo else (0, -12), textcoords="offset points",
                    ha="right" if ultimo else "center", color=INK, fontsize=7.2)
        ax.set_title(nome, fontsize=8.5, pad=4)

    assi[0, 0].annotate("ideale", xy=(2.6, 2.6), xytext=(-3, 4),
                        textcoords="offset points", color=INK2,
                        fontsize=7, rotation=40, ha="center")
    fig.supxlabel("Thread OpenMP", fontsize=8.5)
    fig.supylabel("Speed-up rispetto al sequenziale", fontsize=8.5)
    for est in ("pdf", "png"):
        fig.savefig(uscita / f"scalabilita.{est}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"scritto {uscita/'scalabilita.pdf'} e .png")

    for nome in GRAFI:
        sp = speedup(best, nome)
        print(f"  {nome:<18} speed-up "
              + " ".join(f"{p}t={s:.2f}x" for p, s in zip(THREADS, sp))
              + "   efficienza "
              + " ".join(f"{p}t={s / p:.0%}" for p, s in zip(THREADS, sp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
