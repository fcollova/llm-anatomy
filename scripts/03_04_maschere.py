"""Quattro modi di dire a una parola che cosa non deve guardare.

La maschera del capitolo precedente impedisce di vedere il futuro. Ce ne sono
altre tre di uso comune, che risolvono problemi diversi con lo stesso gesto:
mettere meno infinito in una casella prima della softmax.

Si esegue con `python3 scripts/03_04_maschere.py`
e dura circa 5 secondi.

Da provare: componi finestra scorrevole e causale, e stampa la maschera risultante.
"""

import torch

from llm_anatomy.utils import prepara, titolo, verifica

prepara()

T = 10
punteggi = torch.randn(T, T)


def mostra(nome: str, m: torch.Tensor) -> None:
    print(f"  {nome}")
    for i in range(m.shape[0]):
        print("    " + " ".join("#" if v else "." for v in m[i].tolist()))


def pesi_da(m: torch.Tensor) -> torch.Tensor:
    return punteggi.masked_fill(~m, float("-inf")).softmax(dim=-1)


titolo("Non guardare avanti")
causale = torch.ones(T, T, dtype=torch.bool).tril()
mostra("riga = query, colonna = chiave", causale)
p = pesi_da(causale)
verifica(torch.triu(p, diagonal=1).abs().max().item() == 0.0,
         "nessun peso sopra la diagonale sopravvive al softmax")
print(f"  posizioni visibili: {int(causale.sum())} su {T * T} "
      f"({100 * float(causale.float().mean()):.0f}%)")

titolo("Non guardare il riempimento")
lunghezze = torch.tensor([10, 6, 3])
print(f"  tre sequenze nel batch, lunghezze {lunghezze.tolist()}")
valide = torch.arange(T)[None, :] < lunghezze[:, None]
padding = valide[:, None, :] & causale[None, :, :]
mostra("seconda sequenza (lunghezza 6)", padding[1])
print("  le colonne oltre la lunghezza reale sono spente su tutte le righe:")
print("  senza, il modello attenderebbe a token di riempimento")
p = pesi_da(padding[1])
verifica(p[:, 6:].abs().max().item() == 0.0,
         "nessun peso cade sulle posizioni di riempimento")

titolo("Guardare solo vicino")
finestra = 4
posizioni = torch.arange(T)
scorrevole = causale & ((posizioni[:, None] - posizioni[None, :]) < finestra)
mostra(f"finestra di {finestra} posizioni", scorrevole)
print(f"  visibili: {int(scorrevole.sum())} contro {int(causale.sum())} della")
print("  causale piena: la memoria della KV cache smette di crescere con T")

titolo("Non guardare l'altro documento")
documento = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
per_documento = causale & (documento[:, None] == documento[None, :])
mostra("tre documenti concatenati", per_documento)
print("  senza questa maschera un documento leggerebbe la coda del precedente,")
print("  imparando correlazioni che non esistono")
p = pesi_da(per_documento)
verifica(p[7, :7].abs().max().item() == 0.0,
         "il primo token del terzo documento non vede i due precedenti")

titolo("Si sommano")
composta = padding[1] & scorrevole & per_documento
mostra("riempimento + finestra + documento", composta)
print("  le maschere si compongono con un AND logico. Le righe oltre la")
print("  lunghezza reale della sequenza restano completamente spente: sono")
print("  posizioni di riempimento, e la loro uscita va scartata, non usata.")
righe_reali = composta[:int(lunghezze[1])]
verifica(bool(righe_reali.any(dim=1).all()),
         "ogni riga che corrisponde a un token vero conserva almeno una posizione")
cieche = int((~composta.any(dim=1)).sum())
print(f"  righe interamente spente: {cieche}, tutte oltre la posizione "
      f"{int(lunghezze[1])}")
print("  su una riga cieca il softmax di soli -infinito produce NaN: e' il modo")
print("  piu' comune di rompere un addestramento con le maschere, e si evita")
print("  escludendo quelle posizioni dalla perdita")
