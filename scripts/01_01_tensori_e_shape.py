"""I tensori: i numeri del modello, e la forma che hanno.

Tutto quello che entra, esce e si impara in un modello e' una griglia di numeri
con una forma: si chiama tensore. Un numero solo, una fila, una tabella, una
pila di tabelle. Quasi tutti gli errori che si fanno costruendo un modello
sono errori di forma, e quasi tutta la fatica di leggerne uno e' seguire come
la forma cambia: vale la pena spenderci una pagina prima di cominciare.

Si esegue con `python3 scripts/01_01_tensori_e_shape.py`
e dura circa 5 secondi.

Da provare: cambia B, T e C nella seconda sezione e guarda come cambiano le forme.
"""

import torch

from llm_anatomy.utils import prepara, stampa_shape, titolo, verifica

prepara()

titolo("Un numero, una fila, una tabella")
numero = torch.tensor(3.0)
fila = torch.tensor([1.0, 2.0, 3.0])
tabella = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
for nome, t in (("numero", numero), ("fila", fila), ("tabella", tabella)):
    print(f"  {nome:<8} assi {t.ndim}   forma {tuple(t.shape)}")
print()
print(tabella)

titolo("Una pila di tabelle")
B, T, C = 2, 5, 4    # frasi, caratteri, numeri per carattere
x = torch.arange(B * T * C, dtype=torch.float32).reshape(B, T, C)
stampa_shape("x (B, T, C)", x)
print("\n  la prima frase, x[0]: cinque righe, una per carattere\n")
print(x[0])

titolo("Prendere un pezzo")
stampa_shape("x[0]      una frase", x[0])
stampa_shape("x[0, 2]   un carattere", x[0, 2])
stampa_shape("x[:, -1]  ultimo carattere", x[:, -1])
print(f"  x[0, 2] = {x[0, 2].tolist()}")

titolo("Cambiare forma")
piatto = x.view(B * T, C)
stampa_shape("x.view(B*T, C)", piatto)
verifica(torch.equal(piatto[7], x[1, 2]),
         "la riga 7 della tabella piatta e' il carattere 2 della frase 1")
n_head, d_head = 2, C // 2
teste = x.view(B, T, n_head, d_head).transpose(1, 2)
stampa_shape("(B, T, C) -> (B, h, T, d_h)", teste)
ricomposto = teste.transpose(1, 2).contiguous().view(B, T, C)
verifica(torch.equal(ricomposto, x),
         "divisa in teste e ricomposta, x torna identica")

titolo("Sommare una cosa piccola a una grande")
pos = torch.tensor([100.0, 200.0, 300.0, 400.0])
somma = x + pos
stampa_shape("pos (C,)", pos)
stampa_shape("x + pos", somma)
print(f"  somma[0, 0] = {somma[0, 0].tolist()}")
print(f"  somma[1, 4] = {somma[1, 4].tolist()}")
print("  regola: le forme si allineano da destra,")
print("  (2, 5, 4) e (4,) -> (2, 5, 4)")
verifica(torch.equal(somma[1, 4], x[1, 4] + pos),
         "la stessa fila pos e' stata sommata a ogni carattere di ogni frase")
