"""Cinque minuti di addestramento, e il modello comincia a scrivere.

E' l'unico script lungo del libro. Duemilacinquecento passi su un solo testo --
le ricette dell'Artusi, poco meno di un megabyte -- e alla fine il modello ne
scrive una da solo.
Il checkpoint che produce serve anche ai capitoli seguenti, quindi conviene
lanciarlo e lasciarlo lavorare mentre si legge il resto.

Se lo si interrompe non si perde niente: salva lo stato ogni duecentocinquanta
passi e rilanciandolo riprende da li'.

Si esegue con `python3 scripts/05_02_pretrain_agogpt.py`
e dura circa 5 minuti.

Da provare: riduci i passi a 500 e guarda che testo esce.
"""

import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import prepara_split
from llm_anatomy.model import AgoGPT
from llm_anatomy.sample import genera
from llm_anatomy.train import ConfigAddestramento, addestra
from llm_anatomy.utils import (conta_parametri, figura, numero, percorso,
                               prepara, stile_figure, titolo, verifica)

ap = argparse.ArgumentParser()
ap.add_argument("--max-seconds", type=float, default=None,
                help="ferma il segmento dopo N secondi salvando il checkpoint")
ap.add_argument("--passi", type=int, default=2500)
args, _ = ap.parse_known_args()

prepara()

titolo("Il modello, e quanto testo gli diamo")
tok, dati_train, dati_val = prepara_split()
cfg = AgoGPTConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=4,
                    d_model=192, n_head=6)
modello = AgoGPT(cfg)
print(f"  {cfg.descrizione()}")
print(f"  parametri:        {numero(conta_parametri(modello))}")
print(f"  di cui embedding: {numero(modello.tok_emb.weight.numel())} "
      f"(condivisi con la testa di uscita)")
print(f"  token di addestramento: {numero(len(dati_train))}")

# lr alto e warmup corto: il modello e' piccolo e il rischio di divergere e'
# basso. Su un modello grande questi numeri sarebbero temerari.
tc = ConfigAddestramento(passi=args.passi, batch_size=16, lr=3e-3, warmup=100,
                         intervallo_eval=100, intervallo_checkpoint=250)
print(f"  passi {tc.passi}, batch {tc.batch_size}, contesto {cfg.block_size}, "
      f"lr {tc.lr}")

titolo("Quanto e' ignorante prima di cominciare")
# Un modello che tira a caso fra 93 simboli sbaglia in modo prevedibile: la
# sua perdita vale log(93). Se il modello appena inizializzato si discosta da
# quel numero, qualcosa e' gia' storto prima di cominciare.
perdita_casuale = math.log(tok.vocab_size)
with torch.no_grad():
    x, y = dati_val[:128].unsqueeze(0), dati_val[1:129].unsqueeze(0)
    _, perdita0 = modello(x, y)
print(f"  log(vocab_size) = log({tok.vocab_size}) = {perdita_casuale:.4f}")
print(f"  perdita del modello non addestrato = {perdita0.item():.4f}")
verifica(abs(perdita0.item() - perdita_casuale) < 0.35,
         "il modello inizializzato e' indistinguibile da una scelta uniforme")

titolo("Duemilacinquecento passi")
esito = addestra(modello, dati_train, dati_val, tc, nome="agogpt",
                 max_seconds=args.max_seconds)
print()
print(f"  passo raggiunto: {esito['passo']}/{tc.passi}")
print(f"  perdita di validazione: {esito['loss_val']:.4f}")
print(f"  durata del segmento: {esito['secondi'] / 60:.1f} min")

if not esito["completato"]:
    print("  segmento non conclusivo: rilanciare lo script per proseguire")
    raise SystemExit(0)

verifica(esito["loss_val"] <= 1.70,
         "la perdita di validazione e' scesa sotto la soglia di 1,70")

titolo("Come scende la perdita")
righe = [json.loads(r) for r in
         percorso("outputs", "agogpt_log.jsonl").read_text().splitlines()]
righe = righe[-(tc.passi // tc.intervallo_eval):]
print(f"  {'passo':>6} {'train':>8} {'val':>8} {'ms/passo':>10}")
for r in righe[::5]:
    print(f"  {r['passo']:6d} {r['loss_train']:8.3f} {r['loss_val']:8.3f} "
          f"{r['ms_per_passo']:10.0f}")
ultimo = righe[-1]
print(f"  {ultimo['passo']:6d} {ultimo['loss_train']:8.3f} "
      f"{ultimo['loss_val']:8.3f} {ultimo['ms_per_passo']:10.0f}")
print()
print(f"  perplexity finale: {math.exp(ultimo['loss_val']):.2f} "
      f"su {tok.vocab_size} simboli")

titolo("Le stesse cose, disegnate")
fig, ax = plt.subplots(figsize=(stile_figure(), 2.8))
ax.plot([r["passo"] for r in righe], [r["loss_train"] for r in righe],
        label="addestramento", lw=1)
ax.plot([r["passo"] for r in righe], [r["loss_val"] for r in righe],
        label="validazione", lw=1.6)
ax.axhline(math.log(tok.vocab_size), ls=":", lw=0.8, color="gray")
ax.text(tc.passi * 0.55, math.log(tok.vocab_size) - 0.08,
        "modello uniforme", color="gray", va="top")
ax.set_xlabel("passo")
ax.set_ylabel("perdita (nat/token)")
ax.legend()
figura(fig, "05-02_curva_pretrain")

titolo("Che cosa scrive")
# temperatura 0,8: un po' sotto uno, quindi il modello e' leggermente piu'
# prudente di quanto la sua distribuzione suggerisca. A uno divaga, a 0,5
# ripete.
gen = torch.Generator().manual_seed(1337)
avvio = torch.tensor([tok.encode("800. Risotto")], dtype=torch.long)
uscita = genera(modello, avvio, n_token=400, temp=0.8, k=0, generatore=gen)
testo = tok.decode(uscita[0])
print(f"  avvio: {'800. Risotto'!r}, temperatura 0,8, 400 token")
print()
for riga in testo.split("\n"):
    print(f"  | {riga[:74]}")

parole = [p for p in testo.replace("\n", " ").split(" ") if p.isalpha()]
print()
print(f"  parole alfabetiche generate: {len(parole)}")
media = sum(len(p) for p in parole) / max(1, len(parole))
print(f"  lunghezza media: {media:.1f}")
