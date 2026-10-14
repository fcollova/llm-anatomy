"""Corpus, split e batching causale.

Il corpus sono le ricette de "La scienza in cucina e l'arte di mangiar bene" di
Pellegrino Artusi: 792 ricette, poco meno di un megabyte. Il file sta in data/
gia' ripulito -- senza prefazioni, indici e note dell'edizione -- e ha
un'impronta fissa: i numeri stampati nel libro dipendono dai suoi byte esatti,
e un file diverso darebbe numeri diversi senza dire perche'.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import torch

from llm_anatomy.tokenizer import CharTokenizer
from llm_anatomy.utils import percorso

# L'indirizzo serve solo a chi ha i sorgenti senza la cartella data/: il file
# e' gia' accanto al codice, e finche' c'e' non si scarica niente.
URL_CORPUS = (
    "https://raw.githubusercontent.com/fcollova/llm-anatomy/main/"
    "data/artusi.txt"
)
NOME_CORPUS = "artusi.txt"
SHA256_CORPUS = "86fd4c53baed3478cc3c27c2c6df3d4d5e1fbdf4615bda13cab1835cdd398917"


def scarica_corpus(url: str = URL_CORPUS, nome: str = NOME_CORPUS) -> Path:
    """Scarica il corpus se manca. Restituisce il percorso locale."""
    p = percorso("data", nome)
    if not p.exists():
        print(f"  [download] {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            p.write_bytes(r.read())
    return p


def carica_testo(nome: str = NOME_CORPUS) -> str:
    """Legge il corpus. Quello del libro viene controllato contro l'impronta.

    Un altro file, passato per nome, si legge senza controlli: e' il modo di
    addestrare il modello su un testo proprio.
    """
    dati = scarica_corpus(nome=nome).read_bytes()
    if nome == NOME_CORPUS and hashlib.sha256(dati).hexdigest() != SHA256_CORPUS:
        raise SystemExit(f"data/{nome} non e' il file su cui il libro e' stato "
                         f"costruito: i numeri non coinciderebbero")
    return dati.decode("utf-8")


def tokenizer_corpus(testo: str | None = None) -> CharTokenizer:
    """Il tokenizer del modello, ricostruito dal corpus.

    Deterministico: dipende solo dai caratteri presenti nel corpus, quindi due
    script diversi ottengono la stessa tabella senza scambiarsi file.
    """
    return CharTokenizer.da_testo(testo if testo is not None else carica_testo())


def prepara_split(frazione_train: float = 0.9, testo: str | None = None):
    """Codifica il corpus e lo taglia in due blocchi contigui.

    Lo split e' per posizione, non casuale: cosi' nessuna finestra di contesto
    attraversa il confine fra addestramento e validazione.
    """
    testo = testo if testo is not None else carica_testo()
    tok = tokenizer_corpus(testo)
    dati = torch.tensor(tok.encode(testo), dtype=torch.long)
    n = int(frazione_train * len(dati))
    return tok, dati[:n], dati[n:]


def get_batch(dati: torch.Tensor, batch_size: int, block_size: int,
              generatore: torch.Generator | None = None,
              device: torch.device | str = "cpu"):
    """Estrae un batch causale: y e' x sfalsato di una posizione.

    Le finestre partono da posizioni casuali indipendenti; con un corpus di un
    milione di token e contesto 128 le sovrapposizioni fra batch sono rare e
    non serve tenere traccia di quali finestre sono gia' state viste.
    """
    massimo = len(dati) - block_size - 1
    ix = torch.randint(massimo, (batch_size,), generator=generatore)
    x = torch.stack([dati[i:i + block_size] for i in ix])
    y = torch.stack([dati[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
