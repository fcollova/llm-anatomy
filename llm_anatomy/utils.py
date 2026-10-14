"""Servizi comuni: seed, thread, percorsi, conteggio parametri, tabelle."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 1337

# Numero di thread usato da tutti gli script. E' fissato qui, in un punto solo,
# perche' i tempi misurati siano confrontabili fra uno script e l'altro. Su una
# macchina con meno processi si abbassa con la variabile AGOGPT_THREADS.
THREADS = int(os.environ.get("AGOGPT_THREADS", "8"))


def radice() -> Path:
    """Radice del repository, dedotta dalla posizione di questo file."""
    return Path(__file__).resolve().parent.parent


def percorso(*parti: str) -> Path:
    """Percorso assoluto dentro il repository, creando le cartelle mancanti."""
    p = radice().joinpath(*parti)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def prepara(seed: int = SEED, threads: int = THREADS) -> torch.device:
    """Rende lo script deterministico e stampa l'ambiente in una riga.

    Va chiamata come prima istruzione di ogni script: la riga stampata e' cio'
    che permette al lettore di sapere con quale seed e quanti thread sono stati
    ottenuti i numeri che vede subito sotto.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    dev = dispositivo()
    print(f"[ambiente] torch {torch.__version__}  device={dev}  "
          f"threads={threads}  seed={seed}")
    return dev


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conta_parametri(modulo: torch.nn.Module, solo_addestrabili: bool = False) -> int:
    par = modulo.parameters()
    if solo_addestrabili:
        par = (p for p in modulo.parameters() if p.requires_grad)
    return sum(p.numel() for p in par)


def byte_leggibili(n: float) -> str:
    for unita in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unita == "GiB":
            return f"{n:.1f} {unita}"
        n /= 1024
    return f"{n:.1f} GiB"


def numero(n: float, decimali: int = 0) -> str:
    """Formato italiano: punto per le migliaia, virgola per i decimali."""
    s = f"{n:,.{decimali}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def dec(x: float, decimali: int = 2) -> str:
    """Numero con la virgola decimale, senza toccare nient'altro.

    Serve perche' applicare replace(".", ",") a una stringa gia' composta
    trasforma anche i punti che fanno parte del testo: e' un errore che si e'
    presentato davvero, cambiando il simbolo '.' in ','.
    """
    return f"{x:.{decimali}f}".replace(".", ",")


def titolo(testo: str, larghezza: int = 72) -> None:
    """Intestazione di sezione dell'output. Mai piu' di 80 colonne."""
    print()
    print(testo)
    print("-" * min(larghezza, max(len(testo), 8)))


def tabella(intestazioni: list[str], righe: list[list], allinea: str | None = None) -> None:
    """Stampa una tabella allineata a colonne di larghezza fissa.

    allinea e' una stringa di 'l'/'r' lunga quanto le intestazioni; per default
    la prima colonna a sinistra e le altre a destra, che e' la forma leggibile
    quando le altre colonne sono numeri.
    """
    testo = [[str(c) for c in riga] for riga in righe]
    ncol = len(intestazioni)
    if allinea is None:
        allinea = "l" + "r" * (ncol - 1)
    larghezze = [
        max(len(intestazioni[i]), *(len(r[i]) for r in testo)) if testo else len(intestazioni[i])
        for i in range(ncol)
    ]

    def riga_fmt(celle: list[str]) -> str:
        return "  ".join(
            c.ljust(larghezze[i]) if allinea[i] == "l" else c.rjust(larghezze[i])
            for i, c in enumerate(celle)
        )

    print(riga_fmt(intestazioni))
    print("  ".join("-" * l for l in larghezze))
    for r in testo:
        print(riga_fmt(r))


def verifica(condizione: bool, messaggio: str) -> None:
    """Asserisce un invariante e ne dichiara l'esito su stdout.

    Il libro vive di queste righe: senza la dichiarazione stampata, il
    lettore non sa se il controllo e' stato fatto davvero.
    """
    assert condizione, f"VERIFICA FALLITA: {messaggio}"
    # A capo entro 78 colonne, il testo allineato sotto se stesso: e' quanto
    # entra nel riquadro dell'uscita, e una riga piu' lunga verrebbe spezzata
    # dalla pagina in un punto qualunque.
    import textwrap
    prefisso = "  [verifica superata] "
    print(textwrap.fill(messaggio, width=78, initial_indent=prefisso,
                        subsequent_indent=" " * len(prefisso)))


# Larghezza dello specchio di stampa in pollici: 107,7 mm. Le figure si
# disegnano a questa misura e non piu' larghe. Disegnarle a nove pollici e poi
# ridurle in pagina rimpicciolisce anche le etichette, che a quel punto non si
# leggono: era il difetto della revisione 0.1.
LARGHEZZA_FIGURA = 4.24


def stile_figure() -> float:
    """Prepara matplotlib per la pagina stampata, e dice quanto e' larga.

    Va chiamata prima di creare la figura. Fa tre cose che sulla carta si
    vedono tutte: incorpora i font come TrueType invece che come Type 3, che i
    flussi di stampa trattano male; porta i corpi alla dimensione a cui
    verranno letti, dato che la figura non viene piu' ridotta; e da' a ogni
    serie un tratto proprio oltre al colore, cosi' le curve restano
    distinguibili anche nell'edizione in bianco e nero.
    """
    import matplotlib
    from cycler import cycler

    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "lines.linewidth": 1.1,
        "axes.prop_cycle": (
            cycler(color=["#1F5F63", "#A85B2B", "#6B7076", "#1f4e79"])
            + cycler(linestyle=["-", "--", ":", "-."])
        ),
    })
    return LARGHEZZA_FIGURA


def figura(fig, nome: str) -> Path:
    """Salva una figura in outputs/figures/ e stampa il percorso relativo."""
    p = percorso("outputs", "figures", nome if nome.endswith(".pdf") else nome + ".pdf")
    fig.savefig(p, bbox_inches="tight")
    print(f"  [figura] outputs/figures/{p.name}")
    _mostra_se_notebook(fig)
    return p


def _mostra_se_notebook(fig) -> None:
    """Dentro un notebook la figura compare sotto la cella, oltre che su disco.

    Gli script scelgono il backend Agg, che non mostra niente: da riga di
    comando e' quello che serve, in un notebook la figura resterebbe invisibile.
    Da riga di comando questa funzione non fa e non stampa nulla.
    """
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return
    shell = get_ipython()
    if shell is None or shell.__class__.__name__ != "ZMQInteractiveShell":
        return
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    display(Image(data=buf.getvalue()))


def shape(t: torch.Tensor) -> str:
    """La forma come la scrive Python: (4,) per una fila, () per un numero."""
    return str(tuple(t.shape))


def stampa_shape(etichetta: str, t: torch.Tensor) -> None:
    """Stampa nome, forma e dtype di un tensore, incolonnati."""
    print(f"  {etichetta:<28} {shape(t):<22} {str(t.dtype).replace('torch.', '')}")
