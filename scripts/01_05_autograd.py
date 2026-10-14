"""Il gradiente non e' magia: e' una derivata, e si puo' controllare.

PyTorch calcola da solo di quanto va spostato ogni parametro. Sembra un
oracolo, e invece fa la regola della catena che si studia a scuola. Qui la
rifacciamo a mano tre volte e verifichiamo che i numeri coincidano.

Si esegue con `python3 scripts/01_05_autograd.py`
e dura circa 5 secondi.

Da provare: togli zero_grad() dal ciclo e guarda i gradienti sommarsi.
"""

import torch
import torch.nn.functional as F

from llm_anatomy.utils import prepara, titolo, verifica

prepara()

titolo("Una funzione semplice, derivata a mano")
x = torch.tensor(2.0, requires_grad=True)
y = 3 * x ** 3 + 2 * x
y.backward()
analitica = 9 * 2.0 ** 2 + 2
print(f"  y = 3x^3 + 2x   in x = 2   ->  y = {y.item():.1f}")
print(f"  dy/dx analitica = 9x^2 + 2 = {analitica:.1f}")
print(f"  x.grad          = {x.grad.item():.1f}")
verifica(abs(x.grad.item() - analitica) < 1e-6,
         "autograd riproduce la derivata calcolata a mano")

titolo("La regola della catena, un pezzo alla volta")
a = torch.tensor(1.5, requires_grad=True)
u = torch.sin(a)
v = u ** 2
v.backward()
catena = 2 * torch.sin(torch.tensor(1.5)) * torch.cos(torch.tensor(1.5))
print(f"  v = sin(a)^2   in a = 1,5")
print(f"  dv/du = 2 sin(a) = {2 * torch.sin(torch.tensor(1.5)):.6f}")
print(f"  du/da = cos(a)   = {torch.cos(torch.tensor(1.5)):.6f}")
print(f"  prodotto         = {catena:.6f}")
print(f"  a.grad           = {a.grad.item():.6f}")
verifica(abs(a.grad.item() - catena.item()) < 1e-6,
         "il gradiente e' il prodotto dei due fattori della catena")

titolo("Il gradiente della perdita vera")
V, B = 5, 3
logits = torch.randn(B, V, requires_grad=True)
bersagli = torch.tensor([0, 3, 4])
perdita = F.cross_entropy(logits, bersagli)
perdita.backward()

prob = logits.softmax(dim=-1)
uno_caldo = F.one_hot(bersagli, V).float()
analitico = (prob - uno_caldo) / B

print(f"  perdita: {perdita.item():.4f}")
print("  gradiente analitico (p - y)/B, prima riga:")
print("   " + " ".join(f"{v:8.5f}" for v in analitico[0].tolist()))
print("  gradiente di autograd, prima riga:")
print("   " + " ".join(f"{v:8.5f}" for v in logits.grad[0].tolist()))
verifica(torch.allclose(logits.grad, analitico, atol=1e-6),
         "il gradiente della cross-entropy e' (probabilita - bersaglio)")
print("  e' la forma che rende l'addestramento stabile: il segnale e' grande")
print("  quando il modello sbaglia con sicurezza, piccolo quando ha ragione")

titolo("Perche' bisogna azzerarli ogni volta")
w = torch.tensor(1.0, requires_grad=True)
for passo in range(3):
    perdita_w = (w * 2) ** 2
    perdita_w.backward()
    print(f"  passo {passo}: w.grad = {w.grad.item():6.1f}   "
          f"(contributo del singolo passo: 8w = {8 * w.item():.1f})")
verifica(w.grad.item() == 24.0,
         "senza azzeramento i tre contributi si sommano invece di sostituirsi")

titolo("Staccare un ramo")
p = torch.tensor(3.0, requires_grad=True)
q = (p * 2).detach() * p
q.backward()
print(f"  q = detach(2p) * p   ->  dq/dp = 2p (non 4p) = {p.grad.item():.1f}")
print("  e' cosi' che si ferma il gradiente dove non deve passare, per esempio")
print("  sui bersagli di una distillazione")
