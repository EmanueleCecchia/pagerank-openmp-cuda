#!/usr/bin/env python3
"""Genera le figure di scalabilità della relazione dai CSV in results/<macchina>/.

Produce in relazione/figure/ scalabilita.pdf (speed-up) ed efficienza.pdf
(efficienza parallela, lo speed-up diviso per il numero di thread), vettoriali
per LaTeX, ciascuna con un PNG di controllo. Il riferimento per lo speed-up è
la build sequenziale vera (compilata senza -fopenmp), non la build OpenMP
limitata a un thread.

Un pannello per grafo, con gli assi in comune: otto curve sovrapposte in un
solo grafico si confonderebbero proprio fra 2 e 4 thread, dove si decide la
lettura.  wiki-Vote e' escluso: i suoi tempi sono sotto il millisecondo.

Ogni macchina misurata ha la sua cartella, scritta da run_benchmarks.sh:
results/machine1/bench.csv, results/machine2/bench.csv, ...  Senza argomenti
le prende tutte, in ordine di numero, come "Macchina 1", "Macchina 2", ...
Per sceglierle o rinominarle, una --machine per ciascuna (CSV o cartella):

    python3 tools/plot_results.py --machine 4c/8t results/machine1 \\
                                  --machine 12c/24t results/machine2

Ogni macchina diventa una curva per pannello della figura dell'efficienza, sui
thread che ha misurato: l'efficienza e' normalizzata sul numero di thread, e
rende confrontabili macchine con un numero di core diverso.  Lo speed-up resta
sulla prima.
"""

import argparse
import csv
import re
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
# Una macchina per colore, nell'ordine fisso della palette, e per simbolo:
# le curve restano distinguibili anche stampate in bianco e nero.
STILI = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D")]
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#d4d4d4"


def macchine_misurate(risultati):
    """Le cartelle results/<macchina>/ con un bench.csv, machine2 prima di machine10."""
    cartelle = [d for d in risultati.iterdir() if (d / "bench.csv").is_file()]
    cartelle.sort(key=lambda d: [int(t) if t.isdigit() else t
                                 for t in re.split(r"(\d+)", d.name)])
    return [(etichetta(d.name), d) for d in cartelle]


def etichetta(cartella):
    """machine2 -> "Macchina 2", come nella tabella degli ambienti; altrimenti il nome."""
    m = re.fullmatch(r"machine(\d+)", cartella)
    return f"Macchina {m[1]}" if m else cartella


def carica(path):
    """Minimo per configurazione: l'attivita' di sistema puo' solo rallentare."""
    if path.is_dir():
        path = path / "bench.csv"
    best = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            g = r["graph"].split("/")[-1].replace(".csr", "")
            k = (g, r["build"], r["precision"], int(r["threads"]))
            best[k] = min(best.get(k, float("inf")), float(r["seconds_total"]))
    return best


def threads_misurati(best, nome):
    return sorted(p for (g, build, prec, p) in best
                  if g == nome and build == "omp" and prec == "double")


def speedup(best, nome, threads):
    seq = best[(nome, "seq", "double", 1)]
    return [seq / best[(nome, "omp", "double", p)] for p in threads]


def virgola(x):
    return f"{x:.2f}".replace(".", ",")


def percento(x):
    return f"{x * 100:.0f}%"


def prepara(ax, threads, nome):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xscale("log", base=2)
    # una tacca per ogni numero di thread misurato, ma il numero solo sulle
    # potenze di 2 e sul massimo: 12 e 16 affiancati su scala log si toccano
    ax.set_xticks(threads)
    ax.set_xticklabels([str(p) if p & (p - 1) == 0 or p == threads[-1] else ""
                        for p in threads])
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    ax.set_title(nome, fontsize=8.5, pad=4)


def salva(fig, uscita, nome):
    for est in ("pdf", "png"):
        fig.savefig(uscita / f"{nome}.{est}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    print(f"scritto {uscita / (nome + '.pdf')} e .png")


def figura_speedup(best, uscita):
    fig, assi = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True,
                             layout="constrained")

    for ax, nome in zip(assi.flat, GRAFI):
        prepara(ax, THREADS, nome)
        ax.set_ylim(0, 4.6)
        ax.set_yticks([0, 1, 2, 3, 4])

        # riferimento ideale: linea sottile, tratteggiata, chiaramente non un dato
        ax.plot(THREADS, THREADS, "--", color=INK2, linewidth=0.9,
                dashes=(4, 3), zorder=1)

        sp = speedup(best, nome, THREADS)
        colore, simbolo = STILI[0]
        ax.plot(THREADS, sp, "-" + simbolo, color=colore, linewidth=1.8,
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

    assi[0, 0].annotate("ideale", xy=(2.6, 2.6), xytext=(-3, 4),
                        textcoords="offset points", color=INK2,
                        fontsize=7, rotation=40, ha="center")
    fig.supxlabel("Thread OpenMP", fontsize=8.5)
    fig.supylabel("Speed-up rispetto al sequenziale", fontsize=8.5)
    salva(fig, uscita, "scalabilita")
    plt.close(fig)


def figura_efficienza(macchine, uscita):
    fig, assi = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True,
                             layout="constrained")
    tutti = sorted({p for _, best in macchine for nome in GRAFI
                    for p in threads_misurati(best, nome)})

    for ax, nome in zip(assi.flat, GRAFI):
        prepara(ax, tutti, nome)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(lambda v, _: percento(v))
        ax.axhline(1, linestyle="--", color=INK2, linewidth=0.9,
                   dashes=(4, 3), zorder=1)

        for (etichetta, best), (colore, simbolo) in zip(macchine, STILI):
            threads = threads_misurati(best, nome)
            if not threads or (nome, "seq", "double", 1) not in best:
                continue
            ef = [s / p for s, p in zip(speedup(best, nome, threads), threads)]
            ax.plot(threads, ef, "-" + simbolo, color=colore, linewidth=1.8,
                    markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=3, label=etichetta)
            # un'etichetta per curva: l'efficienza al massimo dei thread, sotto il
            # punto perche' la curva ci arriva scendendo da sinistra.
            ax.annotate(percento(ef[-1]), xy=(threads[-1], ef[-1]),
                        xytext=(4, -12), textcoords="offset points",
                        ha="right", color=INK, fontsize=7.2)

    assi[0, 0].annotate("ideale", xy=(5.66, 1), xytext=(0, 3),
                        textcoords="offset points", color=INK2,
                        fontsize=7, ha="center", va="bottom")
    if len(macchine) > 1:
        # da tutti i pannelli: una macchina puo' non aver misurato ogni grafo
        voci = {}
        for ax in assi.flat:
            for curva, nome in zip(*ax.get_legend_handles_labels()):
                voci.setdefault(nome, curva)
        nomi = [nome for nome, _ in macchine if nome in voci]
        fig.legend([voci[n] for n in nomi], nomi, loc="outside upper center",
                   ncols=len(nomi), frameon=False, fontsize=8)
    fig.supxlabel("Thread OpenMP", fontsize=8.5)
    fig.supylabel("Efficienza parallela", fontsize=8.5)
    salva(fig, uscita, "efficienza")
    plt.close(fig)


def main(argv=None):
    radice = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--machine", nargs=2, action="append",
                        metavar=("NAME", "CSV"),
                        help="una macchina da mettere nella figura dell'efficienza, "
                             "il suo bench.csv o la sua cartella; ripetibile, la "
                             "prima da' anche lo speed-up (default: tutte le "
                             "results/<macchina>/bench.csv)")
    args = parser.parse_args(argv)

    if args.machine:
        elenco = [(nome, Path(path)) for nome, path in args.machine]
    else:
        elenco = macchine_misurate(radice / "results")
        if not elenco:
            parser.error("nessun results/<macchina>/bench.csv: "
                         "lanciare prima MACHINE=... tools/run_benchmarks.sh")
    if len(elenco) > len(STILI):
        parser.error(f"al piu' {len(STILI)} macchine, sceglierle con --machine")
    macchine = [(nome, carica(path)) for nome, path in elenco]
    uscita = radice / "relazione" / "figure"
    uscita.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
    })
    figura_speedup(macchine[0][1], uscita)
    figura_efficienza(macchine, uscita)

    for etichetta, best in macchine:
        print(f"\n{etichetta}")
        for nome in GRAFI:
            threads = threads_misurati(best, nome)
            if not threads:
                continue
            sp = speedup(best, nome, threads)
            print(f"  {nome:<18} speed-up "
                  + " ".join(f"{p}t={s:.2f}x" for p, s in zip(threads, sp))
                  + "   efficienza "
                  + " ".join(f"{p}t={s / p:.0%}" for p, s in zip(threads, sp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
