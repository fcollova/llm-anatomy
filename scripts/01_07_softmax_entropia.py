"""Da punteggi a percentuali, e come si misura una previsione.

Il modello produce punteggi senza scala. La softmax li trasforma in
percentuali che sommano a uno, e la temperatura decide quanto quelle
percentuali siano nette. Poi serve un numero che dica se la previsione era
buona: e' la perdita che vedremo scendere per tutto il capitolo 5.

Si esegue con `python3 scripts/01_07_softmax_entropia.py`
e dura circa 5 secondi.

Da provare: prova temperatura 0,1 e 5,0 sugli stessi logit.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_anatomy.utils import figura, prepara, stile_figure, titolo, verifica

prepara()

logits = torch.tensor([2.0, 1.0, 0.5, -1.0, -2.0])
etichette = ["il", "un", "questo", "gatto", "xkq"]

titolo("Da punteggi a percentuali")
p = logits.softmax(dim=-1)
print(f"  {'simbolo':<10} {'logit':>8} {'exp':>10} {'prob':>8}")
for e, l, pi in zip(etichette, logits.tolist(), p.tolist()):
    print(f"  {e:<10} {l:8.2f} {math.exp(l):10.3f} {pi:8.4f}")
verifica(abs(p.sum().item() - 1.0) < 1e-6, "le probabilita' sommano a uno")

titolo("Aggiungere lo stesso a tutti non cambia niente")
spostati = logits + 100.0
verifica(torch.allclose(spostati.softmax(-1), p, atol=1e-6),
         "sommare una costante a tutti i logit non cambia la distribuzione")
print("  per questo l'implementazione sottrae il massimo prima di")
print("  esponenziare:")
print(f"  exp(100 + 2) = {math.exp(102):.3e} traboccherebbe in float32 gia' a ~88")


def entropia(prob: torch.Tensor) -> float:
    prob = prob[prob > 0]
    return float(-(prob * prob.log()).sum())


titolo("La temperatura: quanto essere decisi")
print(f"  {'T':>6} {'prob del massimo':>18} {'entropia (nat)':>16} "
      f"{'perplexity':>12}")
temperature = [0.25, 0.5, 1.0, 2.0, 5.0]
entropie = []
for t in temperature:
    q = (logits / t).softmax(dim=-1)
    h = entropia(q)
    entropie.append(h)
    print(f"  {t:6.2f} {q.max().item():18.4f} {h:16.4f} {math.exp(h):12.2f}")
print(f"  limite T -> 0: distribuzione concentrata, entropia 0")
print(f"  limite T -> inf: uniforme, entropia log(5) = {math.log(5):.4f}")
verifica(entropie[0] < entropie[-1],
         "l'entropia cresce al crescere della temperatura")

titolo("Misurare quanto una previsione e' incerta")
casi = {
    "quasi certa": torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]),
    "il caso sopra": logits,
    "uniforme": torch.zeros(5),
}
for nome, l in casi.items():
    q = l.softmax(dim=-1)
    h = entropia(q)
    print(f"  {nome:<15} entropia {h:.4f} nat   perplexity {math.exp(h):6.2f}   "
          f"prob max {q.max():.3f}")

titolo("Il numero che diremo perdita")
bersaglio = 0
for nome, l in casi.items():
    q = l.softmax(dim=-1)
    ce = -math.log(float(q[bersaglio]))
    print(f"  {nome:<15} -log p(bersaglio) = {ce:.4f}   "
          f"perplexity {math.exp(ce):8.2f}")
print("  un modello a caso su 93 simboli parte da una perdita di "
      f"{math.log(93):.3f}")
print("  ed e' esattamente il valore da cui parte il modello del capitolo 6")

titolo("Le stesse cose, disegnate")

fig, ax = plt.subplots(2, 1, figsize=(stile_figure(), 4.6))
x = torch.arange(5)
for t in (0.25, 1.0, 5.0):
    ax[0].plot(x, (logits / t).softmax(-1), marker="o", label=f"T = {t}")
ax[0].set_xticks(x, etichette)
ax[0].set_ylabel("probabilità")
ax[0].legend()
tt = torch.linspace(0.1, 5, 100)
ax[1].plot(tt, [entropia((logits / float(t)).softmax(-1)) for t in tt])
ax[1].axhline(math.log(5), ls="--", lw=0.8)
ax[1].set_xlabel("temperatura")
ax[1].set_ylabel("entropia (nat)")
fig.tight_layout()
figura(fig, "01-07_temperatura_entropia")
