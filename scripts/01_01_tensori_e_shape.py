"""Le forme, che sono meta' del mestiere.

Un tensore e' una griglia di numeri con una forma. Quasi tutti gli errori che
si fanno costruendo un modello sono errori di forma, e quasi tutta la fatica di
leggerne uno e' seguire come la forma cambia. Vale la pena spenderci una
pagina prima di cominciare.

Si esegue con `python3 scripts/01_01_tensori_e_shape.py`
e dura circa 5 secondi.

Da provare: cambia B, T e C in cima e guarda come si propagano le forme.
"""

import torch

from llm_anatomy.utils import prepara, stampa_shape, titolo, verifica

prepara()

B, T, C = 2, 5, 4   # batch, tempo (token), canali (d_model)

titolo("Come entra il testo")
x = torch.arange(B * T * C, dtype=torch.float32).reshape(B, T, C)
stampa_shape("x = (B, T, C)", x)
print(f"  elementi: {x.numel()}   memoria: {x.numel() * x.element_size()} byte")
print(f"  stride:   {x.stride()}")

titolo("Cambiare forma non sposta niente")
piatto = x.view(B * T, C)
stampa_shape("view(B*T, C)", piatto)
verifica(piatto.data_ptr() == x.data_ptr(),
         "view condivide la memoria con il tensore di partenza")
piatto[0, 0] = -1.0
verifica(x[0, 0, 0].item() == -1.0,
         "scrivere sulla view modifica anche x")
x[0, 0, 0] = 0.0

titolo("Scambiare due assi, invece, si paga")
xt = x.transpose(1, 2)
stampa_shape("transpose(1, 2)", xt)
print(f"  contiguo: {xt.is_contiguous()}   stride: {xt.stride()}")
try:
    xt.view(B, C * T)
except RuntimeError as e:
    print(f"  view su tensore non contiguo fallisce: {type(e).__name__}")
xtc = xt.contiguous()
stampa_shape("dopo contiguous()", xtc)
verifica(xtc.data_ptr() != xt.data_ptr(),
         "contiguous() ha copiato i dati in un nuovo blocco di memoria")

titolo("Il gesto che si ripete piu' spesso")
n_head, d_head = 2, C // 2
teste = x.view(B, T, n_head, d_head).transpose(1, 2)
stampa_shape("(B, T, C) -> (B, h, T, d_h)", teste)
ricomposto = teste.transpose(1, 2).contiguous().view(B, T, C)
verifica(torch.equal(ricomposto, x),
         "split in teste e ricomposizione riportano al tensore di partenza")

titolo("Sommare una cosa piccola a una grande")
pos = torch.arange(T, dtype=torch.float32).unsqueeze(1).repeat(1, C)
stampa_shape("pos = (T, C)", pos)
somma = x + pos
stampa_shape("x + pos", somma)
print("  regola: le forme si allineano da destra, (2,5,4) e (5,4) -> (2,5,4)")
verifica(torch.equal(somma[0], x[0] + pos) and torch.equal(somma[1], x[1] + pos),
         "lo stesso vettore posizionale e' stato aggiunto a entrambi gli esempi")

titolo("Quando non funziona")
sbagliato = torch.ones(T)
try:
    x + sbagliato
except RuntimeError:
    print(f"  x{tuple(x.shape)} + v{tuple(sbagliato.shape)}: ultimo asse 4 contro 5")
    print("  il broadcasting non allinea (T,) con l'asse dei canali: serve (T, 1)")
corretto = sbagliato.unsqueeze(1)
stampa_shape("dopo unsqueeze(1)", corretto)
stampa_shape("x + v(T,1)", x + corretto)

titolo("Moltiplicare mille matrici insieme")
W = torch.randn(C, 3)
y = x @ W
stampa_shape("x (B,T,C) @ W (C,3)", y)
print("  matmul lascia intatti gli assi a sinistra: solo l'ultimo cambia")
verifica(torch.allclose(y[1, 2], x[1, 2] @ W, atol=1e-6),
         "il prodotto in batch coincide con il prodotto sul singolo token")
