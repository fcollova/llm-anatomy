"""L'attenzione a testa singola, scritta a mano su cinque parole.

Cinque parole, otto numeri per parola, e il meccanismo che permette a ognuna di
guardare le altre. Tutto quello che segue in questo libro e' questo, ripetuto.

Si esegue con `python3 scripts/03_01_attenzione_da_zero.py`
e dura circa 5 secondi.

Da provare: cambia la frase in cima: le percentuali cambiano, la struttura no.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from llm_anatomy.utils import figura, prepara, stile_figure, titolo, verifica

prepara()

# Una frase finta, corta abbastanza da poterla stampare tutta.
frase = ["il", "gatto", "nero", "dorme", "qui"]
T = len(frase)          # quante parole
C = 8                   # quanti numeri descrivono ogni parola in ingresso
d_k = 4                 # quanti ne useremo dentro l'attenzione

x = torch.randn(1, T, C)

titolo("Ogni parola pone tre domande su di se'")
# Le tre matrici hanno forma identica e ingresso identico. Cio' che le rende
# diverse e' soltanto il posto che occupano nella formula che viene dopo.
Wq = nn.Linear(C, d_k, bias=False)   # che cosa sto cercando
Wk = nn.Linear(C, d_k, bias=False)   # che cosa offro a chi cerca
Wv = nn.Linear(C, d_k, bias=False)   # che cosa porto, se mi scelgono
q, k, v = Wq(x), Wk(x), Wv(x)
print(f"  la frase e': {' '.join(frase)}")
print(f"  ogni parola entra come {C} numeri e diventa tre vettori da {d_k}")
print("  si chiamano query, key e value: domanda, offerta, contenuto")

titolo("Chi si allinea con chi")
# Prodotto scalare fra ogni domanda e ogni offerta: grande se puntano nella
# stessa direzione, negativo se puntano in direzioni opposte.
punteggi = q @ k.transpose(-2, -1)
print("  riga = chi guarda, colonna = chi viene guardato\n")
print("         " + " ".join(f"{p:>7}" for p in frase))
for i in range(T):
    riga = " ".join(f"{c:7.3f}" for c in punteggi[0, i].tolist())
    print(f"  {frase[i]:<6} {riga}")
print("\n  sono numeri grezzi: nessuno ha ancora detto quanto contano")

titolo("Perche' si divide per la radice")
prima = punteggi.std()
punteggi = punteggi / math.sqrt(d_k)
print(f"  deviazione standard dei punteggi: {prima:.3f} prima, "
      f"{punteggi.std():.3f} dopo")
print(f"  la divisione e' per sqrt({d_k}) = {math.sqrt(d_k):.3f}")
print("  senza, i punteggi crescono con d_k, la softmax riceve numeri sempre")
print("  piu' distanti fra loro e finisce per dare tutto a una parola sola")

titolo("Nessuno puo' guardare avanti")
# Il modello impara a prevedere la parola successiva: se potesse vedere il
# futuro, il compito sarebbe banale e non imparerebbe niente.
maschera = torch.ones(T, T, dtype=torch.bool).tril()
for i in range(T):
    print(f"  {frase[i]:<6} " + " ".join("guarda" if m else "  .   "
                                          for m in maschera[i].tolist()))
punteggi = punteggi.masked_fill(~maschera, float("-inf"))
print("\n  le caselle chiuse diventano meno infinito, che la softmax")
print("  trasforma in zero")

titolo("Da punteggi a percentuali")
pesi = punteggi.softmax(dim=-1)
for i in range(T):
    riga = " ".join(f"{c:6.1%}" for c in pesi[0, i].tolist())
    print(f"  {frase[i]:<6} {riga}")
verifica(torch.allclose(pesi.sum(dim=-1), torch.ones(1, T), atol=1e-6),
         "ogni riga somma a cento per cento: e' una distribuzione")
verifica(pesi[0, 0, 1:].sum().item() < 1e-9,
         "la prima parola non guarda nulla oltre se stessa")

titolo("Il risultato: ogni parola diventa una miscela")
y = pesi @ v
i = 3
print(f"  la parola {frase[i]!r} esce come miscela di:")
for j in range(i + 1):
    print(f"    {frase[j]:<6} al {pesi[0, i, j]:.1%}")
a_mano = sum(pesi[0, i, j] * v[0, j] for j in range(T))
verifica(torch.allclose(a_mano, y[0, i], atol=1e-6),
         "rifacendo la miscela a mano si ottiene lo stesso vettore")

titolo("Tutto quanto, in quattro righe")
def attenzione(x, Wq, Wk, Wv):
    q, k, v = Wq(x), Wk(x), Wv(x)
    s = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    m = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()
    return s.masked_fill(~m, float("-inf")).softmax(-1) @ v

verifica(torch.allclose(attenzione(x, Wq, Wk, Wv), y, atol=1e-6),
         "la versione compatta fa esattamente quello che abbiamo fatto a mano")
print("  e' l'intera attenzione: tre proiezioni, un prodotto, una maschera,")
print("  una softmax, un altro prodotto")

# La tabella delle percentuali come mappa di colori: piu' scuro, piu' attenzione.
# Le caselle che la maschera chiude restano bianche.
titolo("La stessa tabella, a colori")
from matplotlib.colors import LinearSegmentedColormap
mappa = LinearSegmentedColormap.from_list("tasca", ["#FFFFFF", "#1F5F63"])
p = pesi[0].detach().clone()
p[~torch.ones(T, T, dtype=torch.bool).tril()] = float("nan")
fig, ax = plt.subplots(figsize=(stile_figure() * 0.62, 2.5))
ax.imshow(p, cmap=mappa, vmin=0, vmax=1)
for i in range(T):
    for j in range(i + 1):
        v = float(p[i, j])
        ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center",
                color="white" if v > 0.55 else "black")
ax.set_xticks(range(T), frase)
ax.set_yticks(range(T), frase)
ax.set_xlabel("chi viene guardato")
ax.set_ylabel("chi guarda")
ax.tick_params(length=0)
for lato in ax.spines.values():
    lato.set_visible(False)
figura(fig, "03-01_mappa_attenzione")

