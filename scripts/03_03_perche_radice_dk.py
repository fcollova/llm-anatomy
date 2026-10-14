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

from llm_anatomy.utils import prepara, tabella, titolo, verifica

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
    righe.append([
        d_k,
        f"{grezzi.std():.2f}".replace(".", ","),
        f"{math.sqrt(d_k):.2f}".replace(".", ","),
        f"{scalati.std():.3f}".replace(".", ","),
    ])
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
    righe.append([
        d_k,
        f"{h_senza:.3f}".replace(".", ","),
        f"{h_con:.3f}".replace(".", ","),
        f"{massimo:.3f}".replace(".", ","),
    ])
tabella(["d_k", "entropia senza", "entropia con", "peso massimo senza"], righe)
print("  senza scaling, a d_k = 1024 l'attenzione mette quasi tutta la massa su")
print("  una sola posizione: e' un puntatore, non piu' una media pesata")

titolo("E il modello smette di imparare")
# Confrontare la norma del gradiente rispetto a q non direbbe nulla: il fattore
# di scala moltiplica quel gradiente per costruzione. Cio' che conta e' quanto
# il softmax stesso e' ancora derivabile, cioe' la traccia del suo jacobiano,
# somma di p(1-p): vale zero esatto quando la distribuzione e' un puntatore.
righe = []
for d_k in (16, 64, 256, 1024):
    q = torch.randn(CAMPIONI, T, d_k)
    k = torch.randn(CAMPIONI, T, d_k)
    grezzi = q @ k.transpose(-2, -1)
    for nome, s in (("senza", grezzi), ("con", grezzi / math.sqrt(d_k))):
        p = s.softmax(dim=-1)
        sensibilita = float((p * (1 - p)).sum(dim=-1).mean())
        if nome == "senza":
            senza = sensibilita
        else:
            righe.append([d_k,
                          f"{senza:.4f}".replace(".", ","),
                          f"{sensibilita:.4f}".replace(".", ","),
                          f"{sensibilita / max(senza, 1e-12):.0f}x"])
tabella(["d_k", "sensibilita' senza", "sensibilita' con", "rapporto"], righe)
print("  a d_k = 1024 il softmax non scalato e' praticamente costante rispetto")
print("  ai suoi ingressi: qualunque correzione arrivi dalla perdita, i")
print("  punteggi non si spostano piu' e la testa resta congelata dov'e'")

titolo("Verifiche")
d_k = 1024
q = torch.randn(CAMPIONI, T, d_k)
k = torch.randn(CAMPIONI, T, d_k)
grezzi = q @ k.transpose(-2, -1)
verifica(abs((grezzi / math.sqrt(d_k)).std().item() - 1.0) < 0.05,
         "dopo il fattore la deviazione standard dei punteggi e' circa uno")
verifica(entropia_media(grezzi) < 0.3 * entropia_media(grezzi / math.sqrt(d_k)),
         "senza il fattore l'entropia dell'attenzione crolla sotto un terzo")
