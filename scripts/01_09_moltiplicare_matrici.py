"""Moltiplicare due tabelle: righe per colonne.

La moltiplicazione fra matrici e' un prodotto scalare ripetuto: ogni casella
del risultato e' una riga della prima tabella per una colonna della seconda.
E' l'operazione che il modello fa piu' di ogni altra, ed e' quella che
l'attenzione usa per confrontare tutte le parole con tutte le altre in un
colpo solo.

Si esegue con `python3 scripts/01_09_moltiplicare_matrici.py`
e dura circa 5 secondi.

Da provare: scambia l'ordine, M @ A, e leggi l'errore: dice quali lunghezze non tornano.
"""

import torch

from llm_anatomy.utils import prepara, stampa_shape, titolo, verifica

prepara()

titolo("Righe per colonne, a mano")
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
M = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
P = A @ M
stampa_shape("A", A)
stampa_shape("M", M)
stampa_shape("A @ M", P)
print()
print(P)
riga, colonna = A[0], M[:, 1]
print(f"\n  casella (0, 1): riga 0 di A {riga.tolist()}, colonna 1 di M "
      f"{colonna.tolist()}")
print(f"  1*0 + 2*1 + 3*1 = {torch.dot(riga, colonna).item():.0f}")
verifica(all(P[i, j] == torch.dot(A[i], M[:, j])
             for i in range(2) for j in range(2)),
         "ogni casella e' il prodotto scalare di una riga per una colonna")
print("  le lunghezze interne devono coincidere: (2, 3) @ (3, 2) -> (2, 2)")

titolo("Tutte le coppie in un colpo solo")
T, d = 4, 8
q = torch.randn(T, d)
k = torch.randn(T, d)
punteggi = q @ k.T
print(f"  {T} domande e {T} risposte, ognuna lunga {d} numeri")
print(f"  q @ k.T da' una tabella {tuple(punteggi.shape)}: "
      "una casella per coppia\n")
for i in range(T):
    print("   " + " ".join(f"{v:7.3f}" for v in punteggi[i].tolist()))
a_mano = torch.zeros(T, T)
for i in range(T):
    for j in range(T):
        a_mano[i, j] = torch.dot(q[i], k[j])
verifica(torch.allclose(a_mano, punteggi, atol=1e-6),
         f"le {T * T} caselle coincidono con {T * T} prodotti scalari "
         "fatti a mano")
print("  k.T e' k con righe e colonne scambiate: serve perche' le lunghezze")
print("  interne coincidano, (4, 8) @ (8, 4)")
