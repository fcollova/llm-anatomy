"""Il blocco assemblato, e la cosa che lo rende impilabile.

Attenzione piu' rete, ciascuna con la sua normalizzazione e la sua somma. La
somma e' il pezzo che conta: ogni blocco non riscrive quello che ha ricevuto,
ci aggiunge una correzione, e per questo se ne possono impilare quanti se ne
vuole.

Si esegue con `python3 scripts/03_09_blocco_completo.py`
e dura circa 5 secondi.

Da provare: azzera una sola delle due proiezioni di uscita invece di entrambe.
"""

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import Blk
from llm_anatomy.utils import (conta_parametri, dec, prepara, stampa_shape,
                               tabella, titolo, verifica)

prepara()

cfg = AgoGPTConfig(d_model=192, n_head=6, block_size=64, vocab_size=93)
blocco = Blk(cfg).eval()
x = torch.randn(2, 32, cfg.d_model)

titolo("Il blocco, per intero")
stampa_shape("ingresso", x)
with torch.no_grad():
    y, _ = blocco(x)
stampa_shape("uscita", y)
print(f"  parametri del blocco: {conta_parametri(blocco)}")
for nome, modulo in (("attenzione", blocco.attn), ("FFN", blocco.ffn),
                     ("norme", blocco.norm1)):
    print(f"    {nome:<12} {conta_parametri(modulo)}")

titolo("Azzerandolo, non fa niente")
with torch.no_grad():
    blocco.attn.wo.weight.zero_()
    blocco.ffn.w_down.weight.zero_()
    y0, _ = blocco(x)
verifica(torch.allclose(y0, x, atol=1e-6),
         "azzerando le due proiezioni di uscita, l'ingresso attraversa intatto")
print("  non e' un caso di laboratorio: e' esattamente cio' che accade a")
print("  inizializzazione, dove quelle matrici sono scalate verso zero, e il")
print("  motivo per cui una pila profonda parte comunque stabile")

titolo("Quanto scrive, rispetto a quello che riceve")
blocco = Blk(cfg).eval()
with torch.no_grad():
    a, _ = blocco.attn(blocco.norm1(x))
    dopo_attn = x + a
    f = blocco.ffn(blocco.norm2(dopo_attn))
    finale = dopo_attn + f

def lunghezza(t: torch.Tensor) -> float:
    """La lunghezza media dei vettori, uno per parola."""
    return float(t.norm(dim=-1).mean())


ingresso = lunghezza(x)
righe = [
    ["ingresso", dec(ingresso, 3), "---"],
    ["contributo attenzione", dec(lunghezza(a), 3),
     dec(lunghezza(a) / ingresso, 3)],
    ["contributo FFN", dec(lunghezza(f), 3), dec(lunghezza(f) / ingresso, 3)],
    ["uscita", dec(lunghezza(finale), 3), "---"],
]
tabella(["quantita'", "norma media", "rispetto all'ingresso"], righe)
print("  a inizializzazione i contributi sono piccoli rispetto al flusso:")
print("  ogni blocco corregge, non riscrive")

titolo("Entra e esce quasi uguale")
coseno = torch.nn.functional.cosine_similarity(x, finale, dim=-1).mean()
print(f"  coseno medio: {coseno:.4f}")
verifica(coseno > 0.9,
         "l'uscita del blocco resta quasi allineata al suo ingresso")
print("  e' il motivo per cui ha senso parlare di direzioni nel residual")
print("  stream e leggerle con la testa di uscita, come fa il logit lens")

titolo("Le due disposizioni, davanti a numeri grandi")
# Con un ingresso standard le due varianti si somigliano. La differenza si vede
# moltiplicando per cinque la scala del flusso in entrata: il pre-norm la
# conserva, il post-norm la cancella, perche' l'ultima operazione del blocco e'
# una normalizzazione.
torch.manual_seed(1337)
pre = Blk(cfg, pre_norm=True).eval()
torch.manual_seed(1337)
post = Blk(cfg, pre_norm=False).eval()

righe = []
for fattore in (1.0, 5.0, 20.0):
    xs = x * fattore
    with torch.no_grad():
        y_pre, _ = pre(xs)
        y_post, _ = post(xs)
    righe.append([
        f"{fattore:.0f}x",
        f"{float(xs.norm(dim=-1).mean()):.2f}".replace(".", ","),
        f"{float(y_pre.norm(dim=-1).mean()):.2f}".replace(".", ","),
        f"{float(y_post.norm(dim=-1).mean()):.2f}".replace(".", ","),
    ])
tabella(["scala dell'ingresso", "norma in entrata", "uscita pre-norm",
         "uscita post-norm"], righe)

with torch.no_grad():
    n1, _ = post(x)
    n2, _ = post(x * 20)
verifica(abs(float(n1.norm(dim=-1).mean()) - float(n2.norm(dim=-1).mean())) < 0.5,
         "l'uscita post-norm ha la stessa norma qualunque sia la scala in entrata")
print("  il post-norm consegna al blocco successivo un flusso sempre della")
print("  stessa taglia: comodo per la stabilita' numerica, e fatale per il")
print("  gradiente, perche' il percorso diretto dall'ingresso all'uscita")
print("  viene interrotto a ogni blocco. Lo misura lo script 03-08")
