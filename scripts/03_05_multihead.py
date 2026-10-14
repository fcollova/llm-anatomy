"""Piu' teste: lo stesso calcolo, su fette diverse dello stesso spazio.

Una testa sola vede una relazione alla volta. Ma in una frase ce ne sono molte
in corso insieme -- l'accordo del verbo, il referente del pronome, la parentesi
ancora aperta -- e una sola distribuzione di percentuali non puo' portarle
tutte. La soluzione non e' un meccanismo nuovo: e' lo stesso, ripetuto in
parallelo su porzioni disgiunte dello spazio.

Si esegue con `python3 scripts/03_05_multihead.py`
e dura circa 5 secondi.

Da provare: metti 12 teste invece di 6: i parametri non cambiano.
"""

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import CausalSelfAttention
from llm_anatomy.utils import (conta_parametri, prepara, tabella, titolo,
                               verifica)

prepara()

B = 2       # due frasi in parallelo
T = 16      # sedici parole ciascuna
C = 96      # novantasei numeri per parola
h = 6       # sei teste
d_head = C // h

cfg = AgoGPTConfig(d_model=C, n_head=h, block_size=T, vocab_size=93)
attn = CausalSelfAttention(cfg).eval()
x = torch.randn(B, T, C)

titolo("Dividere non vuol dire aggiungere")
q = attn.wq(x)
# Il gesto e' tutto qui: gli stessi 96 numeri vengono riletti come 6 gruppi da
# 16. view() non sposta un byte, riscrive solo come interpretare la memoria;
# transpose porta le teste davanti, cosi' che il calcolo che segue le tratti
# come se fossero altrettante frasi indipendenti.
qh = q.view(B, T, h, d_head).transpose(1, 2)
print(f"  una parola entra come {C} numeri")
print(f"  esce dalla proiezione ancora come {C} numeri")
print(f"  che rileggiamo come {h} gruppi da {d_head}: {h} x {d_head} = {C}")
print("  nessun numero e' stato aggiunto e nessuno perso")
print(f"\n  forma prima:  {tuple(q.shape)}")
print(f"  forma dopo:   {tuple(qh.shape)}")

titolo("Sei teste costano quanto una")
# La sorpresa: le quattro matrici dell'attenzione restano C x C qualunque sia
# il numero di teste, perche' le teste si spartiscono C invece di moltiplicarlo.
# Aumentare h non e' una spesa: e' una scelta su come dividere lo stesso spazio.
righe = []
for teste in (1, 2, 4, 6, 12):
    c = AgoGPTConfig(d_model=C, n_head=teste, block_size=T, vocab_size=93)
    n = conta_parametri(CausalSelfAttention(c))
    righe.append([teste, C // teste, f"{n:,}".replace(",", ".")])
tabella(["teste", "numeri per testa", "parametri"], righe)
print("  la colonna dei parametri non si muove")
print("  piu' teste = fette piu' sottili, non piu' modello")

titolo("Con una testa sola non e' cambiato niente")
# Controprova: se il multi-head fosse un meccanismo diverso, con h=1 darebbe
# un risultato diverso dall'attenzione del primo script. Non lo da'.
c1 = AgoGPTConfig(d_model=C, n_head=1, block_size=T, vocab_size=93)
a1 = CausalSelfAttention(c1).eval()
with torch.no_grad():
    y1, _ = a1(x)
    q1, k1, v1 = a1.wq(x), a1.wk(x), a1.wv(x)
    s = (q1 @ k1.transpose(-2, -1)) / C ** 0.5
    m = torch.ones(T, T, dtype=torch.bool).tril()
    a_mano = a1.wo(s.masked_fill(~m, float("-inf")).softmax(-1) @ v1)
verifica(torch.allclose(y1, a_mano, atol=1e-5),
         "con una testa sola si torna all'attenzione scritta a mano")

titolo("Con sei teste sono sei conti separati")
# Seconda controprova, piu' forte: si rifanno le sei attenzioni una per una,
# ciascuna sulla propria fetta, e si riconcatenano. Se il risultato coincide,
# allora "multi-head" e' davvero solo questo.
with torch.no_grad():
    y, _ = attn(x)
    q, k, v = (t(x).view(B, T, h, d_head).transpose(1, 2)
               for t in (attn.wq, attn.wk, attn.wv))
    m = torch.ones(T, T, dtype=torch.bool).tril()
    pezzi = []
    for i in range(h):
        s = (q[:, i] @ k[:, i].transpose(-2, -1)) / d_head ** 0.5
        pezzi.append(s.masked_fill(~m, float("-inf")).softmax(-1) @ v[:, i])
    ricomposta = attn.wo(torch.cat(pezzi, dim=-1))
verifica(torch.allclose(y, ricomposta, atol=1e-5),
         f"{h} attenzioni separate, riconcatenate, danno lo stesso risultato")
print("  e' la definizione di multi-head, verificata invece che dichiarata")

titolo("Le teste guardano posti diversi")
# L'ultima riga della matrice dei pesi: dove guarda l'ultima parola, secondo
# ciascuna testa. A pesi casuali le sei distribuzioni si somigliano; dopo
# l'addestramento divergono, ed e' quella divergenza il lavoro delle teste.
with torch.no_grad():
    _, pesi = attn(x, restituisci_pesi=True)
print("  ultima parola della prima frase, dove guarda secondo ogni testa:")
print("  (percentuale sulle prime otto posizioni)\n")
for i in range(h):
    riga = " ".join(f"{p:5.0%}" for p in pesi[0, i, -1, :8].tolist())
    print(f"  testa {i}  {riga}")
