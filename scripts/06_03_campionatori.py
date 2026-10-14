"""Scegliere il carattere da scrivere.

Il modello dice quanto e' probabile ciascun carattere. Trasformare quelle
percentuali in una scelta e' un problema a se', e la manopola che tutti
chiamano "creativita'" e' in realta' un modo di deformare quella distribuzione
prima di pescarci dentro.

Si esegue con `python3 scripts/06_03_campionatori.py`
e dura circa 5 secondi.

Da provare: prova temperatura 0,2 e 1,5 sullo stesso contesto.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import tokenizer_corpus
from llm_anatomy.model import AgoGPT
from llm_anatomy.sample import genera, min_p, temperatura, top_k, top_p
from llm_anatomy.train import carica_checkpoint
from llm_anatomy.utils import (dec, figura, percorso, prepara, stile_figure,
                               tabella, titolo, verifica)

prepara()

file_ckpt = percorso("checkpoints", "agogpt.pt")
if not file_ckpt.exists():
    raise SystemExit("checkpoint assente: eseguire prima 05-02")
stato = carica_checkpoint(file_ckpt)
cfg = AgoGPTConfig(**stato["cfg"])
modello = AgoGPT(cfg)
modello.load_state_dict(stato["modello"])
modello.eval()
tok = tokenizer_corpus()


def simbolo(i) -> str:
    c = tok.itos[int(i)]
    return {"\n": "'\\n'", " ": "'sp'"}.get(c, repr(c))


def descrivi(logits: torch.Tensor) -> tuple[int, float]:
    """Quanti simboli restano in gioco e quanta massa originale conservano."""
    vivi = int((logits > float("-inf")).sum())
    p = logits.softmax(dim=-1)
    return vivi, float(p.max())


titolo("Tre momenti diversi, tre distribuzioni")
contesti = {
    "dopo 'parmigian'": "parmigian",
    "dopo 'Mettete al fuoco un'": "Mettete al fuoco un",
    "a inizio riga": "\n",
}
logit_per_contesto = {}
for nome, testo in contesti.items():
    idx = torch.tensor([tok.encode(testo)])
    with torch.no_grad():
        logits, _ = modello(idx)
    l = logits[0, -1]
    logit_per_contesto[nome] = l
    p = l.softmax(dim=-1)
    entropia = float(-(p * (p + 1e-12).log()).sum())
    top = p.topk(3)
    print(f"  {nome}")
    print(f"    entropia {dec(entropia, 3)} nat, prime tre: "
          + ", ".join(f"{simbolo(i)} {dec(float(v), 3)}"
                      for v, i in zip(top.values, top.indices)))

titolo("Gli stessi numeri, sette modi di tagliarli")
l = logit_per_contesto["a inizio riga"]
righe = []
for nome, filtrati in (
    ("nessun filtro", l),
    ("temperatura 0,7", temperatura(l, 0.7)),
    ("temperatura 1,5", temperatura(l, 1.5)),
    ("greedy (T=0)", temperatura(l, 0.0)),
    ("top-k = 5", top_k(l, 5)),
    ("top-p = 0,9", top_p(l, 0.9)),
    ("min-p = 0,1", min_p(l, 0.1)),
):
    vivi, massa = descrivi(filtrati)
    p = filtrati.softmax(dim=-1)
    top = p.topk(2)
    righe.append([nome, vivi, dec(massa, 3),
                  " ".join(f"{simbolo(i)}" for i in top.indices)])
tabella(["filtro", "simboli in gioco", "prob. del migliore", "primi due"],
        righe)
print(f"  il vocabolario ha {cfg.vocab_size} simboli: la colonna 'in gioco'")
print("  dice quanti ne restano dopo il taglio")

titolo("Perche' un taglio fisso non va bene sempre")
righe = []
for nome in contesti:
    l = logit_per_contesto[nome]
    p = l.softmax(dim=-1)
    entropia = float(-(p * (p + 1e-12).log()).sum())
    vivi_k, _ = descrivi(top_k(l, 5))
    vivi_p, _ = descrivi(top_p(l, 0.9))
    vivi_m, _ = descrivi(min_p(l, 0.1))
    righe.append([nome[:26], dec(entropia, 2), vivi_k, vivi_p, vivi_m])
tabella(["contesto", "entropia", "top-k=5", "top-p=0,9", "min-p=0,1"], righe)
print("  top-k lascia sempre cinque simboli, che il modello sia sicuro o no.")
print("  min-p ne lascia pochi quando la distribuzione e' appuntita e molti")
print("  quando e' piatta: la soglia e' relativa al migliore, non assoluta")

vivi_min = [r[4] for r in righe]
verifica(max(vivi_min) > min(vivi_min),
         "min-p cambia il numero di candidati a seconda del contesto")
verifica(all(r[2] == 5 for r in righe),
         "top-k ne lascia esattamente cinque in ogni contesto")

titolo("L'effetto sul testo")
avvio = torch.tensor([tok.encode("Risotto coi funghi\n\n")])
impostazioni = [
    ("greedy", dict(temp=0.0)),
    ("temperatura 0,8", dict(temp=0.8)),
    ("temperatura 1,4", dict(temp=1.4)),
    ("temp 1,0 + top-k 10", dict(temp=1.0, k=10)),
    ("temp 1,0 + min-p 0,1", dict(temp=1.0, minp=0.1)),
]
for nome, kwargs in impostazioni:
    g = torch.Generator().manual_seed(7)
    uscita = genera(modello, avvio.clone(), 90, generatore=g, **kwargs)
    testo = tok.decode(uscita[0, avvio.shape[1]:]).replace("\n", " / ")
    print(f"  {nome}")
    print(f"    {testo[:70]}")

titolo("In pratica")
print("  greedy si ripete, perche' sceglie sempre lo stesso continuo piu'")
print("  probabile; la temperatura alta senza tagli produce simboli rari e")
print("  parole inesistenti. I tagli servono a togliere la coda della")
print("  distribuzione senza appiattire cio' che resta, ed e' il motivo per")
print("  cui in produzione si combina quasi sempre una temperatura moderata")
print("  con un taglio, non l'una o l'altro")

# Le percentuali "a inizio riga", dalla piu' alta alla piu' bassa, e dove le
# taglia ciascun filtro: il top-k dopo un numero fisso di simboli, il top-p
# quando la somma arriva a 0,9, il min-p sotto un decimo del migliore.
titolo("I tagli, disegnati")
prob = logit_per_contesto["a inizio riga"].softmax(dim=-1).sort(descending=True).values
n_k = 5
n_p = int((prob.cumsum(0) - prob < 0.9).sum())
soglia_minp = 0.1 * float(prob[0])
n_minp = int((prob >= soglia_minp).sum())
fig, ax = plt.subplots(figsize=(stile_figure(), 2.4))
ax.bar(range(len(prob)), prob.tolist(), width=0.8, color="#6B7076")
tagli = ((n_k, f"top-k tiene {n_k}", "-", 0.95),
         (n_minp, f"min-p tiene {n_minp}", "--", 0.75),
         (n_p, f"top-p tiene {n_p}", ":", 0.55))
for n, nome, stile, altezza in tagli:
    ax.axvline(n - 0.5, color="#1F5F63", ls=stile, lw=1)
    ax.text(n + 0.8, float(prob[0]) * altezza, nome, color="#1F5F63", va="top")
ax.axhline(soglia_minp, color="#A85B2B", ls="--", lw=0.8)
ax.set_xlabel("simboli, dal più probabile al meno probabile")
ax.set_ylabel("probabilità")
ax.set_xlim(-1, len(prob))
print(f"  top-k tiene {n_k} simboli, min-p {n_minp}, top-p {n_p}")
figura(fig, "06-03_tagli")

