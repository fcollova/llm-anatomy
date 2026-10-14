"""Il prodotto scalare: la domanda "quanto siete d'accordo?".

Due liste di numeri della stessa lunghezza si moltiplicano posizione per
posizione e si sommano. Esce un numero solo, ed e' grande quando le due liste
puntano nella stessa direzione. E' l'unica operazione che serve capire per
capire l'attenzione, e vale la pena passarci qualche minuto adesso.

Si esegue con `python3 scripts/01_02_prodotto_scalare.py`
e dura circa 5 secondi.

Da provare: porta d a 4096 e guarda l'ultima tabella: l'oscillazione arriva a 64.
"""

import torch
import torch.nn.functional as F

from llm_anatomy.utils import prepara, titolo, verifica

prepara()

titolo("Quattro coppie, quattro risposte")
a = torch.tensor([1.0, 0.0])          # una freccia che punta a destra
casi = {
    "stessa direzione": torch.tensor([2.0, 0.0]),
    "ad angolo retto": torch.tensor([0.0, 3.0]),
    "direzione opposta": torch.tensor([-2.0, 0.0]),
    "a quarantacinque gradi": torch.tensor([1.0, 1.0]),
}
print(f"  la prima freccia e' sempre {a.tolist()}, cioe' punta a destra\n")
print(f"  {'la seconda':<24} {'':<12} {'a.b':>7} {'gradi':>7}")
for nome, b in casi.items():
    coseno = F.cosine_similarity(a, b, dim=0).clamp(-1, 1)
    gradi = torch.rad2deg(torch.acos(coseno))
    print(f"  {nome:<24} {str(b.tolist()):<12} {torch.dot(a, b):7.2f} {gradi:7.0f}")
verifica(torch.dot(a, casi["ad angolo retto"]).item() == 0.0,
         "due direzioni ad angolo retto danno esattamente zero")
print("  positivo se sono d'accordo, zero se non c'entrano nulla,")
print("  negativo se si contraddicono")

titolo("Attenzione pero': misura anche la lunghezza")
lungo = torch.tensor([10.0, 10.0])
corto = torch.tensor([0.1, 0.1])
for nome, b in (("lunga", lungo), ("corta", corto)):
    coseno = F.cosine_similarity(a, b, dim=0)
    print(f"  freccia {nome:<6} lunghezza {b.norm():6.2f}   "
          f"a.b {torch.dot(a, b):7.2f}   angolo uguale (coseno {coseno:.3f})")
print("  stessa direzione, prodotti scalari lontanissimi: il numero cresce")
print("  se le frecce si allungano, anche quando l'accordo non cambia")

titolo("Tutte le coppie in un colpo solo")
T, d = 4, 8
q = torch.randn(T, d)
k = torch.randn(T, d)
punteggi = q @ k.T
print(f"  {T} domande e {T} risposte, ognuna lunga {d} numeri")
print(f"  q @ k.T da' una tabella {tuple(punteggi.shape)}: una casella per coppia\n")
for i in range(T):
    print("   " + " ".join(f"{v:7.3f}" for v in punteggi[i].tolist()))
a_mano = torch.zeros(T, T)
for i in range(T):
    for j in range(T):
        a_mano[i, j] = torch.dot(q[i], k[j])
verifica(torch.allclose(a_mano, punteggi, atol=1e-6),
         f"le {T * T} caselle coincidono con {T * T} prodotti scalari fatti a mano")
print("  una moltiplicazione fra matrici e' questo: tutti i confronti insieme")

titolo("Piu' numeri, punteggi piu' grandi")
print(f"  {'lunghezza':>10} {'media':>9} {'quanto oscilla':>16} {'radice':>9}")
for dim in (8, 32, 128, 512):
    x = torch.randn(4096, dim)
    y = torch.randn(4096, dim)
    p = (x * y).sum(dim=1)
    print(f"  {dim:10d} {p.mean():9.3f} {p.std():16.3f} {dim ** 0.5:9.3f}")
print("  la media resta zero, ma l'oscillazione cresce come la radice della")
print("  lunghezza: sommare 512 termini invece di 8 da' numeri piu' grandi")
print("  anche se ogni termine e' rimasto della stessa taglia")
