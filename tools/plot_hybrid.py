#!/usr/bin/env python3
"""Genera la figura della versione ibrida della relazione dai CSV in results/.

Produce in relazione/figure/ quote.pdf, con un PNG di controllo: per ogni
grafo lo speed-up della versione ibrida rispetto alla GPU sola, al variare
della quota di archi data alla CPU (-s), in doppia precisione.  Sopra 1
dividere il lavoro conviene, sotto conviene lasciarlo tutto alla GPU.

Un pannello per grafo, con gli assi in comune, come nelle figure di
plot_results.py, di cui riprende grafi, colori e stile.  Una curva per
configurazione, cioe' per GPU: di default la GTX 1050 Ti della Macchina 1, la
GTX 1080 Ti della Macchina 2 e le due RTX 2080 Ti della Macchina 3, sul
collegamento x8 e su quello x16.  Quattro curve al piu' per pannello, quanti
sono i colori della palette.  Per sceglierle o rinominarle, una --config per
ciascuna (CSV o cartella), che prendono i colori nell'ordine:

    python3 tools/plot_hybrid.py --config "RTX 2080 Ti" results/machine2/rtx2080ti

Sopra il massimo di ogni curva, dove supera 1,1, c'e' il suo speed-up, nel
colore della curva.  Per ogni quota si prende il minimo sulle ripetizioni,
come nelle tabelle.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from plot_results import GRAFI, GRID, INK, INK2, STILI, plt, salva, virgola

# Nell'ordine della legenda, ciascuna con il suo stile di STILI.  Il giallo,
# il meno leggibile come testo, va alla x8, che con le misure attuali non
# supera mai 1,1 e quindi non ha etichette.
CONFIG = [
    ("GTX 1050 Ti (Macchina 1)", "results/machine1", 0),
    ("GTX 1080 Ti (Macchina 2)", "results/machine2/gtx1080ti", 1),
    ("RTX 2080 Ti x8 (Macchina 3)", "results/machine3/x8", 3),
    ("RTX 2080 Ti x16 (Macchina 3)", "results/machine3/x16", 2),
]
MAX_CURVE = len(STILI)


def carica(path):
    """Tempo minimo della build ibrida, per grafo e quota, in doppia precisione."""
    if path.is_dir():
        path = path / "bench.csv"
    best = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["build"] != "hybrid" or r["precision"] != "double":
                continue
            k = (Path(r["graph"]).stem, float(r["cpu_share"]))
            best[k] = min(best.get(k, float("inf")), float(r["seconds_total"]))
    return best


def etichetta_massimi(ax, massimi):
    """Scrive sopra ogni massimo (quota, speed-up, colore) il suo speed-up.
    Due massimi alla stessa quota e quasi alla stessa altezza avrebbero le
    etichette una sull'altra: le si affianca, quella del piu' basso a sinistra
    del punto, l'altra a destra.  Un fondo bianco interrompe le curve che
    passano sotto un'etichetta, come in plot_results.py."""
    massimi = sorted(massimi)   # per quota, poi per speed-up
    for q, s, colore in massimi:
        vicini = [m for m in massimi if m[0] == q and abs(math.log2(m[1] / s)) < 0.5]
        if len(vicini) < 2:
            dx, ha = 0, "center"
        elif (q, s, colore) == vicini[0]:
            dx, ha = -1, "right"
        else:
            dx, ha = 1, "left"
        ax.annotate(f"{virgola(s)}×", xy=(q, s), xytext=(dx, 5),
                    textcoords="offset points", ha=ha, color=colore,
                    fontsize=7, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))


def figura_quote(config, uscita):
    fig, assi = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True,
                             layout="constrained")
    for ax, nome in zip(assi.flat, GRAFI):
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for lato in ("top", "right"):
            ax.spines[lato].set_visible(False)
        ax.set_title(nome, fontsize=8.5, pad=4)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "", "0,5", "", "1"])
        # log: guadagni e perdite rispetto alla GPU sola restano simmetrici
        ax.set_yscale("log", base=2)
        ax.set_ylim(1 / 16, 4)
        ax.set_yticks([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2])
        ax.set_yticklabels(["1/16", "1/8", "1/4", "1/2", "1", "2"])
        ax.minorticks_off()
        # riferimento: la GPU sola, linea sottile tratteggiata
        ax.axhline(1, linestyle="--", color=INK2, linewidth=0.9, dashes=(4, 3), zorder=1)

        massimi = []
        for etichetta, best, (colore, simbolo) in config:
            quote = sorted(s for g, s in best if g == nome)
            if 0.0 not in quote:
                continue
            sp = [best[(nome, 0.0)] / best[(nome, s)] for s in quote]
            ax.plot(quote, sp, "-" + simbolo, color=colore, linewidth=1.6,
                    markersize=3.8, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=3, label=etichetta)
            # un'etichetta per curva, e solo dove dividere il lavoro rende:
            # il massimo, che e' il dato riportato nella tabella
            i = max(range(len(sp)), key=sp.__getitem__)
            if sp[i] > 1.1:
                massimi.append((quote[i], sp[i], colore))
        etichetta_massimi(ax, massimi)

    assi[0, 0].annotate("GPU sola", xy=(1.0, 1), xytext=(0, 3),
                        textcoords="offset points", color=INK2, fontsize=7,
                        ha="right", va="bottom")
    voci, nomi = assi[0, 0].get_legend_handles_labels()
    # quattro voci su una riga allargherebbero la figura oltre i pannelli, e
    # in pagina la rimpicciolirebbero: oltre tre, due colonne
    fig.legend(voci, nomi, loc="outside upper center",
               ncols=len(nomi) if len(nomi) <= 3 else 2,
               frameon=False, fontsize=8)
    fig.supxlabel("Quota degli archi alla CPU (-s)", fontsize=8.5)
    fig.supylabel("Speed-up rispetto alla GPU sola", fontsize=8.5)
    salva(fig, uscita, "quote")
    plt.close(fig)


def main(argv=None):
    radice = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", nargs=2, action="append", metavar=("NAME", "CSV"),
                        help="una GPU da mettere nella figura, il suo bench.csv o "
                             f"la sua cartella; ripetibile, al piu' {MAX_CURVE} volte "
                             "(default: la GPU della Macchina 1, la GTX 1080 Ti "
                             "della Macchina 2 e le due GPU della Macchina 3)")
    args = parser.parse_args(argv)

    if args.config and len(args.config) > MAX_CURVE:
        parser.error(f"al piu' {MAX_CURVE} configurazioni")
    elenco = ([(nome, Path(p), STILI[i]) for i, (nome, p) in enumerate(args.config)]
              if args.config
              else [(nome, radice / p, STILI[i]) for nome, p, i in CONFIG])
    config = [(nome, carica(p), stile) for nome, p, stile in elenco]
    uscita = radice / "relazione" / "figure"
    uscita.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
    })
    figura_quote(config, uscita)
    return 0


if __name__ == "__main__":
    sys.exit(main())
