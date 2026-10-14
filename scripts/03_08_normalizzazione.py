"""Tenere i numeri in scala, e dove mettere l'operazione che lo fa.

Impilando decine di blocchi i numeri tendono a crescere o a spegnersi. Si
rimedia normalizzandoli, e il punto interessante non e' come si normalizza ma
dove: la stessa operazione, spostata di due righe, decide se una pila profonda
si addestra o no.

Si esegue con `python3 scripts/03_08_normalizzazione.py`
e dura circa 10 secondi.

Da provare: porta i blocchi da 24 a 48 e rifai il confronto sui gradienti.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import Blk, RMSNorm
from llm_anatomy.utils import (conta_parametri, figura, prepara, stile_figure,
                               tabella, titolo, verifica)

prepara()

titolo("Due modi di normalizzare")
d = 256
x = torch.randn(4, 16, d)
ln = nn.LayerNorm(d, bias=False)
rms = RMSNorm(d)
y_ln, y_rms = ln(x), rms(x)
print(f"  media dell'ingresso:  {x.mean():+.4f}")
print(f"  media dopo LayerNorm: {y_ln.mean():+.4f}")
print(f"  media dopo RMSNorm:   {y_rms.mean():+.4f}")
print(f"  scarto massimo fra le due uscite: {(y_ln - y_rms).abs().max():.4f}")
print(f"  parametri: LayerNorm {conta_parametri(ln)}, "
      f"RMSNorm {conta_parametri(rms)}")

titolo("Quando coincidono")
x0 = x - x.mean(dim=-1, keepdim=True)
verifica(torch.allclose(ln(x0), rms(x0), atol=1e-5),
         "senza media da sottrarre, LayerNorm e RMSNorm fanno la stessa cosa")
print("  e' l'osservazione da cui nasce RMSNorm: la sottrazione della media")
print("  non serviva, e costa una passata in piu' sui dati")

titolo("Ventiquattro blocchi, e il gradiente")
N_BLOCCHI = 24
cfg = AgoGPTConfig(d_model=128, n_head=4, block_size=64, vocab_size=93,
                   act="gelu")
risultati = {}
for etichetta, pre in (("pre-norm", True), ("post-norm", False)):
    torch.manual_seed(1337)
    blocchi = nn.ModuleList([Blk(cfg, pre_norm=pre) for _ in range(N_BLOCCHI)])
    x = torch.randn(2, 64, cfg.d_model, requires_grad=True)
    h = x
    for b in blocchi:
        h, _ = b(h)
    h.pow(2).mean().backward()
    norme = [float(b.attn.wq.weight.grad.norm()) for b in blocchi]
    risultati[etichetta] = norme

righe = []
for i in (0, 5, 11, 17, 23):
    pre, post = risultati["pre-norm"][i], risultati["post-norm"][i]
    rapporto = pre / max(post, 1e-30)
    righe.append([i, f"{pre:.3e}", f"{post:.3e}", f"{rapporto:.0f}x"])
tabella(["blocco", "pre-norm", "post-norm", "rapporto"], righe)

rapporto_medio = (sum(risultati["pre-norm"]) / sum(risultati["post-norm"]))
print(f"  il gradiente post-norm e' in media {rapporto_medio:.0f} volte piu'")
print("  piccolo di quello pre-norm, a ogni profondita'")
verifica(rapporto_medio > 100,
         "attraverso 24 blocchi post-norm il gradiente si attenua di "
         "oltre due ordini di grandezza")
print("  la differenza non e' nella distribuzione lungo la pila -- in")
print("  entrambi gli schemi i primi blocchi ricevono piu' segnale degli")
print("  ultimi -- ma nella scala assoluta: nel post-norm ogni blocco")
print("  normalizza il flusso residuo, e il percorso diretto dall'ingresso")
print("  all'uscita non esiste piu'. E' la ragione del warmup nel paper del")
print("  2017 e dell'instabilita' che si osservava sopra i dodici blocchi")

titolo("Come cresce il flusso")
attivazioni = {}
for etichetta, pre in (("pre-norm", True), ("post-norm", False)):
    torch.manual_seed(1337)
    blocchi = nn.ModuleList([Blk(cfg, pre_norm=pre) for _ in range(N_BLOCCHI)])
    h = torch.randn(2, 64, cfg.d_model)
    serie = []
    with torch.no_grad():
        for b in blocchi:
            h, _ = b(h)
            serie.append(float(h.norm(dim=-1).mean()))
    attivazioni[etichetta] = serie
righe = []
for i in (0, 11, 23):
    righe.append([i,
                  f"{attivazioni['pre-norm'][i]:.2f}".replace(".", ","),
                  f"{attivazioni['post-norm'][i]:.2f}".replace(".", ",")])
tabella(["blocco", "pre-norm", "post-norm"], righe)
print("  nel pre-norm la norma cresce lungo la pila, perche' ogni blocco")
print("  aggiunge al flusso residuo senza che nulla lo rinormalizzi: e' il")
print("  motivo per cui serve una norma finale prima della testa di uscita")

fig, ax = plt.subplots(2, 1, figsize=(stile_figure(), 4.6))
for etichetta, norme in risultati.items():
    ax[0].semilogy(norme, marker="o", ms=3, lw=1, label=etichetta)
ax[0].set_xlabel("blocco")
ax[0].set_ylabel("norma del gradiente su Wq")
ax[0].legend()
for etichetta, serie in attivazioni.items():
    ax[1].plot(serie, marker="o", ms=3, lw=1, label=etichetta)
ax[1].set_xlabel("blocco")
ax[1].set_ylabel("norma media delle attivazioni")
ax[1].legend()
figura(fig, "03-08_prenorm_postnorm")
