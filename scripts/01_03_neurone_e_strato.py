"""Uno strato: la cosa di cui il modello e' fatto quasi per intero.

Un neurone somma un prodotto scalare e un numero. Uno strato e' una fila di
neuroni che guardano lo stesso ingresso. Il modello di questo libro e' fatto
quasi tutto di strati: le tre proiezioni dell'attenzione sono strati, la rete
del blocco sono due strati, la testa di uscita e' uno strato.

Si esegue con `python3 scripts/01_03_neurone_e_strato.py`
e dura circa 5 secondi.

Da provare: aggiungi un quarto neurone e conta i parametri prima di eseguire.
"""

import torch
import torch.nn as nn

from llm_anatomy.utils import (conta_parametri, prepara, stampa_shape, titolo,
                               verifica)

prepara()

d_in = 4     # quanti numeri entrano
d_out = 3    # quanti neuroni, quindi quanti numeri escono
B = 5        # quante righe si elaborano insieme

# Un neurone fa due cose: un prodotto scalare fra il suo vettore di pesi e
# l'ingresso, piu' un numero fisso. Nient'altro.
titolo("Un neurone, a mano")
x = torch.randn(d_in)
w = torch.randn(d_in)
b = torch.tensor(0.5)
y = torch.dot(w, x) + b
print(f"  x  {[round(v, 3) for v in x.tolist()]}")
print(f"  w  {[round(v, 3) for v in w.tolist()]}")
print(f"  b  {b.item():.3f}")
print(f"  y = w.x + b = {y.item():.4f}")

titolo("Uno strato: molti neuroni sullo stesso ingresso")
strato = nn.Linear(d_in, d_out, bias=True)
stampa_shape("weight", strato.weight)
stampa_shape("bias", strato.bias)
print(f"  parametri: {conta_parametri(strato)} = "
      f"{d_out}*{d_in} pesi + {d_out} bias")

X = torch.randn(B, d_in)
Y = strato(X)
stampa_shape("X (B, d_in)", X)
stampa_shape("Y (B, d_out)", Y)

titolo("E' davvero un prodotto scalare per neurone")
a_mano = torch.stack([
    torch.stack([torch.dot(strato.weight[j], X[i]) + strato.bias[j]
                 for j in range(d_out)])
    for i in range(B)
])
verifica(torch.allclose(a_mano, Y, atol=1e-6),
         "il calcolo neurone per neurone coincide con nn.Linear")
verifica(torch.allclose(X @ strato.weight.T + strato.bias, Y, atol=1e-6),
         "e coincide con la forma matriciale X W^T + b")

titolo("Il piu' delle volte, oggi, il bias non c'e'")
senza = nn.Linear(d_in, d_out, bias=False)
print(f"  con bias:   {conta_parametri(strato)} parametri")
print(f"  senza bias: {conta_parametri(senza)} parametri")
print("  su d=4096 il risparmio e' trascurabile: si toglie perche' non serve,")
print("  non per risparmiare memoria")
