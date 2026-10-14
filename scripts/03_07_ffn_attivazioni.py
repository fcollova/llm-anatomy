"""La parte del blocco dove stanno quasi tutti i parametri.

Dopo l'attenzione ogni parola passa da sola in una piccola rete: sale in una
dimensione piu' grande, attraversa una non linearita', torna giu'. E' meta' del
modello in numero di parametri, ed e' anche il posto dove la scelta della
funzione fa la differenza.

Si esegue con `python3 scripts/03_07_ffn_attivazioni.py`
e dura circa 5 secondi.

Da provare: lascia d_ff a 4*d con SwiGLU e guarda i parametri crescere del 50%.
"""

import torch
import torch.nn.functional as F

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import FFN, SwiGLU
from llm_anatomy.utils import (conta_parametri, dec, numero, prepara, stampa_shape,
                               tabella, titolo, verifica)

prepara()

titolo("Tre funzioni a confronto")
x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
righe = []
for v in x.tolist():
    t = torch.tensor(v)
    righe.append([dec(v, 1), dec(F.relu(t), 4), dec(F.gelu(t), 4),
                  dec(F.silu(t), 4)])
tabella(["x", "ReLU", "GELU", "SiLU"], righe, allinea="rrrr")
print("  ReLU azzera di netto, GELU e SiLU lasciano passare un poco di")
print("  negativo: la derivata non e' mai esattamente zero, e un neurone")
print("  spento puo' tornare a vivere")

titolo("Quello che le distingue davvero")
for nome, fn in (("ReLU", F.relu), ("GELU", F.gelu), ("SiLU", F.silu)):
    t = torch.tensor(0.0, requires_grad=True)
    fn(t).backward()
    t2 = torch.tensor(-2.0, requires_grad=True)
    fn(t2).backward()
    print(f"  {nome:<6} f'(0) = {t.grad.item():7.4f}   "
          f"f'(-2) = {t2.grad.item():8.5f}")

titolo("La versione classica: due matrici")
d = 512
cfg = AgoGPTConfig(d_model=d, n_head=8, act="gelu", vocab_size=93)
ffn = FFN(cfg)
print(f"  d_model {d}, d_ff {cfg.d_ff} = 4 x {d}")
x = torch.randn(2, 8, d)
stampa_shape("ingresso", x)
stampa_shape("dopo la salita", ffn.w_up(x))
stampa_shape("uscita", ffn(x))
print(f"  parametri: {conta_parametri(ffn)} = 2 x {d} x {cfg.d_ff}")

titolo("SwiGLU: tre matrici, e una porta")
cfg_sw = AgoGPTConfig(d_model=d, n_head=8, act="swiglu", vocab_size=93)
sw = SwiGLU(cfg_sw)
print(f"  d_ff scelta dalla configurazione: {cfg_sw.d_ff}")
print(f"  8/3 x {d} = {dec(8 * d / 3, 1)}, arrotondato al multiplo di 32")
stampa_shape("porta", sw.w_gate(x))
stampa_shape("salita", sw.w_up(x))
stampa_shape("uscita", sw(x))

titolo("Perche' otto terzi")
righe = []
for nome, n_matrici, fattore in (("FFN classica", 2, 4.0),
                                 ("SwiGLU a 4d", 3, 4.0),
                                 ("SwiGLU a 8/3 d", 3, 8 / 3)):
    d_ff = int(fattore * d)
    par = n_matrici * d * d_ff
    classica = 2 * d * 4 * d
    righe.append([nome, n_matrici, d_ff, numero(par), dec(par / classica)])
tabella(["variante", "matrici", "d_ff", "parametri", "rispetto alla classica"],
        righe)
verifica(abs(3 * (8 / 3) - 2 * 4) < 1e-9,
         "tre matrici di larghezza 8/3 d contano quanto due di larghezza 4d")
print("  la sostituzione e' quindi gratuita in parametri e in calcolo: si")
print("  paga solo una moltiplicazione elemento per elemento in piu'")

titolo("Sul nostro modello")
righe = []
for act in ("relu", "gelu", "swiglu"):
    c = AgoGPTConfig(d_model=192, n_head=6, act=act, vocab_size=93)
    modulo = SwiGLU(c) if act == "swiglu" else FFN(c)
    righe.append([act, c.d_ff, conta_parametri(modulo)])
tabella(["attivazione", "d_ff", "parametri della FFN"], righe)
