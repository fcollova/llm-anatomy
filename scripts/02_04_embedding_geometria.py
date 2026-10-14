"""Ogni simbolo diventa un punto, e i punti finiscono vicini per un motivo.

Il modello non lavora sui simboli ma su vettori: una tabella traduce ogni
simbolo in una lista di numeri, e quella tabella si impara insieme a tutto il
resto. Il risultato e' che simboli usati in modo simile finiscono in posti
vicini, senza che nessuno gliel'abbia detto.

Si esegue con `python3 scripts/02_04_embedding_geometria.py`
e dura circa 5 secondi.

Da provare: confronta i vicini di 'a' con quelli di 'z'.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import tokenizer_corpus
from llm_anatomy.model import AgoGPT
from llm_anatomy.train import carica_checkpoint
from llm_anatomy.utils import (dec, figura, percorso, prepara, stampa_shape,
                               stile_figure, tabella, titolo, verifica)

prepara()

file_ckpt = percorso("checkpoints", "agogpt.pt")
if not file_ckpt.exists():
    raise SystemExit("checkpoint assente: eseguire prima 05-02")

stato = carica_checkpoint(file_ckpt)
cfg = AgoGPTConfig(**stato["cfg"])
modello = AgoGPT(cfg)
modello.load_state_dict(stato["modello"])
modello.eval()
tok = tokenizer_corpus()

titolo("Una riga di numeri per simbolo")
E = modello.tok_emb.weight.detach()
stampa_shape("tok_emb.weight", E)
print(f"  passo di addestramento del checkpoint: {stato['passo']}")
print(f"  {cfg.vocab_size} simboli in uno spazio a {cfg.d_model} dimensioni")
print(f"  norma media: {E.norm(dim=1).mean():.4f}   "
      f"minima: {E.norm(dim=1).min():.4f}   "
      f"massima: {E.norm(dim=1).max():.4f}")

titolo("Chi finisce vicino a chi")
En = F.normalize(E, dim=1)
sim = En @ En.T
sim.fill_diagonal_(-2.0)      # nessun simbolo e' il vicino di se stesso


def nome(i: int) -> str:
    c = tok.itos[i]
    return {"\n": "\\n", " ": "spazio"}.get(c, repr(c))


righe = []
for c in "aeiouzQ,.3":
    if c not in tok.stoi:
        continue
    i = tok.stoi[c]
    vicini = sim[i].topk(4).indices.tolist()
    righe.append([nome(i)] + [f"{nome(j)} {dec(float(sim[i, j]))}"
                              for j in vicini])
tabella(["simbolo", "1o vicino", "2o vicino", "3o vicino", "4o vicino"], righe)

titolo("Vocali con vocali, cifre con cifre")
gruppi = {
    "vocali minuscole": "aeiou",
    "consonanti minuscole": "bcdfghlmnprstv",
    "maiuscole": "ABCDEFGHILMNOPRSTUV",
    "punteggiatura": ".,;:!?'",
}
indici = {g: [tok.stoi[c] for c in cs if c in tok.stoi]
          for g, cs in gruppi.items()}


def coseno_medio(sim, da: list[int], verso: list[int]) -> float:
    """La somiglianza media fra due gruppi di simboli."""
    return float(torch.tensor([[sim[i, j] for j in verso if j != i]
                               for i in da]).mean())


righe, scarti = [], []
for gruppo, idx in indici.items():
    resto = [i for i in range(cfg.vocab_size) if i not in idx]
    dentro = coseno_medio(sim, idx, idx)
    fuori = coseno_medio(sim, idx, resto)
    scarti.append(dentro - fuori)
    righe.append([gruppo, len(idx), dec(dentro, 3), dec(fuori, 3),
                  ("+" if dentro >= fuori else "") + dec(dentro - fuori, 3)])
tabella(["gruppo", "n", "coseno interno", "coseno esterno", "scarto"], righe)
verifica(all(s > 0 for s in scarti),
         "ogni gruppo e' piu' coeso al suo interno che verso il resto del "
         "vocabolario")

titolo("La stessa cosa, disegnata")
centrato = E - E.mean(dim=0, keepdim=True)
# Decomposizione completa, non troncata: con 93 simboli in 192 dimensioni
# costa nulla, e le quote di varianza sono cosi' calcolate sul totale vero.
U, S, Vt = torch.linalg.svd(centrato, full_matrices=False)
proiezione = centrato @ Vt[:2].T
quote = (S ** 2 / (S ** 2).sum())
for k in (2, 6, 20):
    print(f"  varianza spiegata dalle prime {k:2d} componenti: "
          f"{100 * quote[:k].sum():5.1f}%")
print(f"  componenti necessarie per il 90%: "
      f"{int((quote.cumsum(0) < 0.90).sum()) + 1} su {min(E.shape)}")
print("  la proiezione qui sotto e' quindi un'ombra: utile per riconoscere i")
print("  gruppi, non per misurare distanze")

fig, ax = plt.subplots(figsize=(stile_figure(), 3.6))
for i in range(cfg.vocab_size):
    c = tok.itos[i]
    etichetta = {"\n": "\\n", " ": "sp"}.get(c, c)
    colore = ("tab:red" if c.isupper() else "tab:blue" if c.islower()
              else "tab:green" if c.isdigit() else "tab:gray")
    ax.text(proiezione[i, 0].item(), proiezione[i, 1].item(), etichetta, color=colore, ha="center", va="center")
ax.set_xlim(proiezione[:, 0].min() - 0.3, proiezione[:, 0].max() + 0.3)
ax.set_ylim(proiezione[:, 1].min() - 0.3, proiezione[:, 1].max() + 0.3)
ax.set_xlabel("prima componente")
ax.set_ylabel("seconda componente")
ax.set_title("rosso: maiuscole, blu: minuscole, verde: cifre, grigio: resto")
figura(fig, "02-04_embedding_pca")

titolo("Prima dell'addestramento non c'era niente")
casuale = AgoGPT(cfg).tok_emb.weight.detach()
Cn = F.normalize(casuale, dim=1)
simc = Cn @ Cn.T
simc.fill_diagonal_(-2.0)
vocali = indici["vocali minuscole"]
dentro_c = coseno_medio(simc, vocali, vocali)
dentro_a = coseno_medio(sim, vocali, vocali)
print(f"  coseno medio fra vocali, modello casuale:    {dentro_c:+.4f}")
print(f"  coseno medio fra vocali, modello addestrato: {dentro_a:+.4f}")
verifica(dentro_a > dentro_c + 0.05,
         "la struttura non c'era all'inizializzazione: e' stata appresa")
