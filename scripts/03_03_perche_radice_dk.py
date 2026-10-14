"""Che cosa succede senza quella radice quadrata.

Nel capitolo 2 abbiamo visto che i punteggi crescono con la lunghezza dei
vettori. Qui si guarda l'effetto sulla softmax, e si scopre che il problema non
e' estetico: senza il correttivo il modello smette proprio di imparare.

Si esegue con `python3 scripts/03_03_perche_radice_dk.py`
e dura circa 5 secondi.

Da provare: prova d_k = 1024 e misura l'entropia dell'attenzione.
"""

import math

import torch

from llm_anatomy.utils import dec, prepara, tabella, titolo, verifica

prepara()

T, CAMPIONI = 64, 256


def entropia_media(punteggi: torch.Tensor) -> float:
    p = punteggi.softmax(dim=-1)
    return float(-(p * (p + 1e-12).log()).sum(dim=-1).mean())


titolo("Piu' numeri, punteggi piu' sparpagliati")
righe = []
for d_k in (4, 16, 64, 256, 1024):
    q = torch.randn(CAMPIONI, T, d_k)
    k = torch.randn(CAMPIONI, T, d_k)
    grezzi = q @ k.transpose(-2, -1)
    scalati = grezzi / math.sqrt(d_k)
    righe.append([d_k, dec(grezzi.std()), dec(math.sqrt(d_k)),
                  dec(scalati.std(), 3)])
tabella(["d_k", "dev. std grezza", "sqrt(d_k)", "dev. std scalata"], righe)
print("  la deviazione grezza segue sqrt(d_k); dopo la divisione resta uno")
print("  a ogni dimensione, che e' esattamente lo scopo del fattore")

titolo("La softmax si irrigidisce")
entropia_uniforme = math.log(T)
print(f"  entropia di una distribuzione uniforme su {T} posizioni: "
      f"{entropia_uniforme:.4f} nat")
print()
righe = []
for d_k in (4, 16, 64, 256, 1024):
    q = torch.randn(CAMPIONI, T, d_k)
    k = torch.randn(CAMPIONI, T, d_k)
    grezzi = q @ k.transpose(-2, -1)
    h_senza = entropia_media(grezzi)
    h_con = entropia_media(grezzi / math.sqrt(d_k))
    massimo = float(grezzi.softmax(-1).max(dim=-1).values.mean())
    righe.append([d_k, dec(h_senza, 3), dec(h_con, 3), dec(massimo, 3)])
tabella(["d_k", "entropia senza", "entropia con", "peso massimo senza"], righe)
print("  senza scaling, a d_k = 1024 l'attenzione mette quasi tutta la massa")
print("  su una sola posizione: e' un puntatore, non piu' una media pesata")

titolo("E il modello smette di imparare")
# Confrontare la norma del gradiente rispetto a q non direbbe nulla: il fattore
# di scala moltiplica quel gradiente per costruzione. Cio' che conta e' quanto
# il softmax stesso e' ancora derivabile, cioe' la traccia del suo jacobiano,
# somma di p(1-p): vale zero esatto quando la distribuzione e' un puntatore.


def sensibilita(p: torch.Tensor) -> float:
    """Quanto si muovono le percentuali se i punteggi si muovono di poco."""
    return float((p * (1 - p)).sum(dim=-1).mean())


righe = []
for d_k in (16, 64, 256, 1024):
    q = torch.randn(CAMPIONI, T, d_k)
    k = torch.randn(CAMPIONI, T, d_k)
    grezzi = q @ k.transpose(-2, -1)
    senza = sensibilita(grezzi.softmax(dim=-1))
    con = sensibilita((grezzi / math.sqrt(d_k)).softmax(dim=-1))
    rapporto = con / max(senza, 1e-12)
    righe.append([d_k, dec(senza, 4), dec(con, 4), f"{rapporto:.0f}x"])
tabella(["d_k", "sensibilita' senza", "sensibilita' con", "rapporto"], righe)
print("  a d_k = 1024 il softmax non scalato e' praticamente costante")
print("  rispetto ai suoi ingressi: qualunque correzione arrivi dalla")
print("  perdita, i punteggi non si spostano piu' e la testa resta congelata")
print("  dov'e'")

titolo("Verifiche")
d_k = 1024
q = torch.randn(CAMPIONI, T, d_k)
k = torch.randn(CAMPIONI, T, d_k)
grezzi = q @ k.transpose(-2, -1)
verifica(abs((grezzi / math.sqrt(d_k)).std().item() - 1.0) < 0.05,
         "dopo il fattore la deviazione standard dei punteggi e' circa uno")
verifica(entropia_media(grezzi) < 0.3 * entropia_media(grezzi / math.sqrt(d_k)),
         "senza il fattore l'entropia dell'attenzione crolla sotto un terzo")
