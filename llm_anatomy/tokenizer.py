"""Tokenizer a caratteri (il filo conduttore) e involucro per un BPE addestrato in casa.

Il modello del libro usa il tokenizer a caratteri:
vocabolario di 93 simboli, nessun file esterno, nessun download. Il BPE serve
solo negli script del capitolo 2, per confrontare le due segmentazioni.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Corrispondenza biunivoca fra caratteri del corpus e interi.

    Volutamente banale: rende visibile che un tokenizer e' solo una tabella, e
    che tutto cio' che il modello vede sono indici in quella tabella.
    """

    def __init__(self, caratteri: list[str]):
        self.itos = list(caratteri)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def da_testo(cls, testo: str) -> "CharTokenizer":
        return cls(sorted(set(testo)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def salva(self, percorso: str | Path) -> None:
        Path(percorso).write_text(
            json.dumps({"itos": self.itos}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def carica(cls, percorso: str | Path) -> "CharTokenizer":
        dati = json.loads(Path(percorso).read_text(encoding="utf-8"))
        return cls(dati["itos"])


def addestra_bpe(testo: str, vocab_size: int = 512, min_frequency: int = 2):
    """Addestra un BPE con la libreria `tokenizers`, senza toccare la rete.

    Restituisce l'oggetto Tokenizer. Import locale: gli script T0 che non usano
    il BPE non devono pagare la dipendenza.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,          # le barre sporcano la cattura dell'output
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([testo], trainer=trainer)
    return tok
