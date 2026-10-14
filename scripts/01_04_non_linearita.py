"""Due strati senza niente in mezzo sono uno strato solo.

E' la ragione per cui esistono le funzioni di attivazione, e si dimostra in
mezza pagina: si prendono due strati lineari, si calcola la matrice che fa
esattamente lo stesso lavoro, e si verifica che coincidano. Poi si mette una
funzione in mezzo e quella matrice smette di esistere.

Si esegue con `python3 scripts/01_04_non_linearita.py`
e dura circa 5 secondi.

Da provare: sostituisci la ReLU con una funzione lineare: la controprova torna a passare.
"""

import torch
import torch.nn as nn

from llm_anatomy.utils import prepara, stampa_shape, titolo, verifica

prepara()

d_in, d_mid, d_out, B = 6, 16, 4, 128

L1 = nn.Linear(d_in, d_mid)
L2 = nn.Linear(d_mid, d_out)
X = torch.randn(B, d_in)

titolo("Due strati, uno dopo l'altro")
h = L1(X)
y = L2(h)
stampa_shape("X", X)
stampa_shape("dopo L1", h)
stampa_shape("dopo L2", y)

titolo("Una matrice sola fa lo stesso lavoro")
W = L2.weight @ L1.weight
b = L2.bias + L2.weight @ L1.bias
stampa_shape("W = W2 W1", W)
stampa_shape("b = b2 + W2 b1", b)
y_eq = X @ W.T + b
scarto = (y - y_eq).abs().max().item()
print(f"  scarto massimo fra i due percorsi: {scarto:.3e}")
verifica(torch.allclose(y, y_eq, atol=1e-5),
         "un solo strato riproduce esattamente la coppia di strati")
print(f"  parametri usati: {d_in * d_mid + d_mid + d_mid * d_out + d_out} "
      f"contro {d_in * d_out + d_out} necessari")

titolo("Con qualcosa in mezzo, non piu'")
y_nl = L2(torch.relu(L1(X)))
scarto_nl = (y_nl - y_eq).abs().max().item()
print(f"  scarto contro la matrice equivalente: {scarto_nl:.3f}")

# La miglior approssimazione lineare possibile: soluzione ai minimi quadrati
# sull'insieme dei punti. Se la rete con ReLU fosse lineare, il residuo
# sarebbe nullo a meno dell'errore numerico.
X1 = torch.cat([X, torch.ones(B, 1)], dim=1)   # piu' il termine fisso
soluzione = torch.linalg.lstsq(X1, y_nl).solution
residuo = (X1 @ soluzione - y_nl).pow(2).mean().sqrt().item()
varianza = y_nl.std().item()
print(f"  residuo della migliore approssimazione lineare: {residuo:.4f}")
print(f"  deviazione standard dell'uscita:                {varianza:.4f}")
print(f"  quota di segnale non spiegabile linearmente:    "
      f"{100 * residuo / varianza:.1f}%")
verifica(residuo > 0.05 * varianza,
         "nessuna trasformazione lineare riproduce la rete con ReLU")

titolo("Controprova")
soluzione2 = torch.linalg.lstsq(X1, y).solution
residuo2 = (X1 @ soluzione2 - y).pow(2).mean().sqrt().item()
print(f"  residuo: {residuo2:.3e}  (zero a meno dell'errore numerico)")
verifica(residuo2 < 1e-4,
         "senza attivazione i minimi quadrati ricostruiscono la rete esatta")
