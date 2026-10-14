"""Il ciclo completo, su un problema che si disegna.

Quattro righe dentro un ciclo: guarda, sbaglia, chiedi in che direzione
muoverti, muoviti. E' lo stesso ciclo che nel capitolo 5 addestrera' il
modello vero, qui applicato a un problema abbastanza piccolo da vederlo tutto.

Si esegue con `python3 scripts/01_06_training_loop.py`
e dura circa 10 secondi.

Da provare: porta il tasso di apprendimento a 0,5: la rete non converge piu'.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from llm_anatomy.utils import (conta_parametri, figura, prepara, stile_figure,
                               titolo, verifica)

prepara()

N, PASSI = 1024, 2000

# Problema: due spirali intrecciate, separabili solo da un confine curvo.
def spirali(n: int):
    t = torch.linspace(0.4, 9.0, n // 2)     # una spirale e mezza per classe
    r = t / 3
    x0 = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1)
    x1 = torch.stack([r * torch.cos(t + 3.14159), r * torch.sin(t + 3.14159)], dim=1)
    X = torch.cat([x0, x1]) + 0.04 * torch.randn(n, 2)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    return X, y


X, y = spirali(N)
Xv, yv = spirali(256)
print(f"  dati: {tuple(X.shape)} esempi, due classi, confine non lineare")


titolo("Il ciclo, in quattro righe")
def addestra(modello, nome: str) -> list[float]:
    ott = torch.optim.Adam(modello.parameters(), lr=0.02)
    perdita_fn = nn.CrossEntropyLoss()
    storia = []
    for passo in range(PASSI):
        logits = modello(X)                # 1. la previsione
        perdita = perdita_fn(logits, y)    # 2. quanto sbaglia
        ott.zero_grad(set_to_none=True)    #    via i gradienti vecchi
        perdita.backward()                 # 3. i gradienti nuovi
        ott.step()                         # 4. i pesi si spostano
        if (passo + 1) % 100 == 0:
            storia.append(perdita.item())
        if (passo + 1) % 500 == 0:
            with torch.no_grad():
                acc = (modello(Xv).argmax(1) == yv).float().mean().item()
            print(f"  {nome:<8} passo {passo + 1:5d}  "
                  f"perdita {perdita.item():.4f}  accuratezza val {acc:.3f}")
    return storia


titolo("Un modello che non ce la fa")
lineare = nn.Linear(2, 2)
print(f"  parametri: {conta_parametri(lineare)}")
storia_lin = addestra(lineare, "lineare")

titolo("Lo stesso problema, con una piega")
rete = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                     nn.Linear(32, 2))
print(f"  parametri: {conta_parametri(rete)}")
storia_rete = addestra(rete, "rete")

titolo("Quanto e' bravo, alla fine")
with torch.no_grad():
    acc_lin = (lineare(Xv).argmax(1) == yv).float().mean().item()
    acc_rete = (rete(Xv).argmax(1) == yv).float().mean().item()
print(f"  lineare: {acc_lin:.3f}")
print(f"  rete:    {acc_rete:.3f}")
verifica(acc_rete > acc_lin + 0.2,
         "la rete non lineare supera nettamente il modello lineare")

fig, ax = plt.subplots(2, 1, figsize=(stile_figure(), 4.6))
passi = [100 * (i + 1) for i in range(len(storia_lin))]
ax[0].plot(passi, storia_lin, label=f"lineare (acc {acc_lin:.2f})")
ax[0].plot(passi, storia_rete, label=f"rete (acc {acc_rete:.2f})")
ax[0].set_xlabel("passo")
ax[0].set_ylabel("perdita di addestramento")
ax[0].legend()
ax[1].scatter(X[y == 0, 0], X[y == 0, 1], s=4)
ax[1].scatter(X[y == 1, 0], X[y == 1, 1], s=4)
ax[1].set_aspect("equal")
fig.tight_layout()
figura(fig, "01-06_curva_perdita")
