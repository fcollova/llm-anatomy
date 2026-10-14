"""Dire al modello in che ordine stanno le parole, senza dirgli dove sono.

L'attenzione da sola non sa nulla dell'ordine: mescolando le parole otterrebbe
lo stesso risultato. RoPE risolve ruotando i vettori di un angolo che dipende
dalla posizione, con una conseguenza sorprendente: il punteggio fra due parole
finisce per dipendere solo da quanto sono lontane, non da dove si trovano.

Si esegue con `python3 scripts/02_06_rope.py` e dura circa 5 secondi.

Da provare: porta la distanza a 500 e guarda il punteggio smettere di decrescere.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_anatomy.model import applica_rope, frequenze_rope
from llm_anatomy.utils import (dec, figura, prepara, stampa_shape, stile_figure,
                               tabella, titolo, verifica)

prepara()

d_head, T = 32, 256
cos, sin = frequenze_rope(d_head, T)
stampa_shape("cos", cos)
stampa_shape("sin", sin)
print(f"  ogni coppia di canali ruota di un angolo proporzionale alla "
      f"posizione")

titolo("Due parole, spostate insieme lungo la frase")
q = torch.randn(1, 1, 1, d_head)
k = torch.randn(1, 1, 1, d_head)
print(f"  prodotto scalare senza rotazione: {(q * k).sum().item():.6f}")
print()

distanza = 5
righe = []
for m in (0, 10, 50, 100, 200):
    n = m + distanza
    qm = applica_rope(q, cos[m:m + 1], sin[m:m + 1])
    kn = applica_rope(k, cos[n:n + 1], sin[n:n + 1])
    righe.append([m, n, n - m, dec((qm * kn).sum().item(), 6)])
tabella(["pos. query", "pos. chiave", "differenza", "prodotto scalare"], righe)

valori = []
for m in range(0, T - distanza):
    qm = applica_rope(q, cos[m:m + 1], sin[m:m + 1])
    kn = applica_rope(k, cos[m + distanza:m + distanza + 1],
                      sin[m + distanza:m + distanza + 1])
    valori.append((qm * kn).sum().item())
scarto = max(valori) - min(valori)
print(f"  su tutte le {len(valori)} posizioni possibili "
      f"a distanza {distanza},")
print(f"  lo scarto fra massimo e minimo e' {scarto:.3e}")
verifica(scarto < 1e-4,
         "il punteggio dipende solo da m-n, non dalle posizioni assolute")

titolo("Conta solo quanto sono lontane")
righe = []
for dist in (0, 1, 2, 8, 32, 128):
    q0 = applica_rope(q, cos[0:1], sin[0:1])
    kd = applica_rope(k, cos[dist:dist + 1], sin[dist:dist + 1])
    righe.append([dist, dec((q0 * kd).sum().item(), 4)])
tabella(["distanza", "prodotto scalare"], righe)
print("  a distanza zero il punteggio e' quello dei vettori non ruotati:")
print("  RoPE non aggiunge informazione, la ridistribuisce fra gli angoli")

titolo("Il metodo del 2017 non si comporta cosi'")
print("  i posizionali sinusoidali si sommano all'embedding e occupano quindi")
print("  parte del residual stream. RoPE ruota solo query e chiave dentro")
print("  l'attenzione: l'embedding resta intatto, e la posizione non consuma")
print("  capacita' del flusso residuo")

titolo("Ruotare non allunga niente")
qm = applica_rope(q, cos[42:43], sin[42:43])
verifica(abs(qm.norm().item() - q.norm().item()) < 1e-5,
         "la rotazione conserva la norma dei vettori")

profili = {}
for dist_max in (T - 1,):
    q0 = applica_rope(q, cos[0:1], sin[0:1])
    serie = []
    for dist in range(dist_max):
        kd = applica_rope(k, cos[dist:dist + 1], sin[dist:dist + 1])
        serie.append((q0 * kd).sum().item())
    profili["punteggio"] = serie

fig, ax = plt.subplots(2, 1, figsize=(stile_figure(), 4.6))
ax[0].plot(profili["punteggio"], lw=1)
ax[0].set_xlabel("distanza fra query e chiave")
ax[0].set_ylabel("prodotto scalare")
ax[1].plot(valori, lw=1)
ax[1].set_xlabel("posizione della query")
ax[1].set_ylabel("prodotto scalare")
ax[1].set_ylim(min(valori) - 1, max(valori) + 1)
fig.tight_layout()
figura(fig, "02-06_rope")
