"""La stessa cosa, scritta dalla libreria.

L'attenzione della sezione precedente esiste gia' dentro PyTorch, in una
versione piu' veloce. Qui si verifica che le due diano lo stesso risultato,
perche' e' l'unico modo di usare quella veloce senza smettere di sapere cosa fa.

Si esegue con `python3 scripts/03_02_verifica_sdpa.py`
e dura circa 5 secondi.

Da provare: togli is_causal e guarda le due versioni divergere.
"""

import math
import time

import torch
import torch.nn.functional as F

from llm_anatomy.utils import (byte_leggibili, prepara, stampa_shape, tabella,
                               titolo, verifica)

prepara()

B, h, T, d_k = 2, 4, 64, 32

q = torch.randn(B, h, T, d_k)
k = torch.randn(B, h, T, d_k)
v = torch.randn(B, h, T, d_k)
stampa_shape("q, k, v", q)


def attenzione_a_mano(q, k, v, causale=True):
    """Le cinque righe del capitolo precedente, senza scorciatoie."""
    punteggi = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causale:
        T = q.shape[-2]
        maschera = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        punteggi = punteggi.masked_fill(~maschera, float("-inf"))
    return punteggi.softmax(dim=-1) @ v


titolo("Il caso che useremo sempre")
mia = attenzione_a_mano(q, k, v, causale=True)
libreria = F.scaled_dot_product_attention(q, k, v, is_causal=True)
scarto = (mia - libreria).abs().max().item()
print(f"  scarto massimo:  {scarto:.3e}")
print(f"  scarto relativo: {scarto / libreria.abs().max().item():.3e}")
verifica(torch.allclose(mia, libreria, atol=1e-6),
         "le due implementazioni coincidono entro 1e-6 nel caso causale")

titolo("Con una maschera scritta a mano")
maschera = torch.rand(T, T) > 0.3
maschera = maschera.tril()
maschera[:, 0] = True          # il primo token resta sempre visibile
punteggi = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
mia2 = punteggi.masked_fill(~maschera, float("-inf")).softmax(-1) @ v
libreria2 = F.scaled_dot_product_attention(q, k, v, attn_mask=maschera)
print(f"  scarto massimo: {(mia2 - libreria2).abs().max().item():.3e}")
verifica(torch.allclose(mia2, libreria2, atol=1e-6),
         "coincidono anche con una maschera arbitraria")

titolo("Senza maschera")
mia3 = attenzione_a_mano(q, k, v, causale=False)
libreria3 = F.scaled_dot_product_attention(q, k, v)
verifica(torch.allclose(mia3, libreria3, atol=1e-6),
         "coincidono anche senza maschera")

titolo("E allora perche' usarla")
def cronometra(fn, n=50):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n

righe = []
for lunghezza in (128, 512, 1024):
    qq = torch.randn(1, h, lunghezza, d_k)
    kk = torch.randn(1, h, lunghezza, d_k)
    vv = torch.randn(1, h, lunghezza, d_k)
    n = 20 if lunghezza < 1024 else 5
    t_mano = cronometra(lambda: attenzione_a_mano(qq, kk, vv), n)
    t_lib = cronometra(lambda: F.scaled_dot_product_attention(qq, kk, vv, is_causal=True), n)
    matrice = h * lunghezza * lunghezza * 4
    righe.append([
        lunghezza,
        f"{1000 * t_mano:.2f}".replace(".", ","),
        f"{1000 * t_lib:.2f}".replace(".", ","),
        f"{t_mano / t_lib:.1f}x".replace(".", ","),
        byte_leggibili(matrice),
    ])
tabella(["T", "a mano (ms)", "libreria (ms)", "rapporto",
         "matrice dei pesi"], righe)
print("  l'ultima colonna e' la memoria che la versione a mano deve allocare e")
print("  che la libreria non alloca mai: cresce con il quadrato di T, ed e' la")
print("  ragione per cui a contesto lungo la differenza smette di essere una")
print("  questione di velocita' e diventa una questione di fattibilita'")

titolo("Che cosa non si vede piu'")
print("  i pesi di attenzione non sono piu' osservabili: per guardarli")
print("  si torna alla versione a mano, attivandola con")
print("  restituisci_pesi=True in llm_anatomy.model.CausalSelfAttention")
