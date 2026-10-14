"""Il modello intero: dal numero del carattere alla previsione.

Tutti i pezzi sono pronti. Qui si mettono in fila e si verifica che il risultato
sia un modello vero: che la forma non cambi mai dentro la pila, che non sappia
nulla del futuro, e che appena costruito sia esattamente ignorante quanto deve.

Si esegue con `python3 scripts/03_10_modello_completo.py`
e dura circa 5 secondi.

Da provare: cambia n_layer e verifica che le tre verifiche passino comunque.
"""

import math

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import AgoGPT
from llm_anatomy.utils import (conta_parametri, prepara, stampa_shape,
                               tabella, titolo, verifica)

prepara()

cfg = AgoGPTConfig(vocab_size=93, block_size=128, n_layer=4, d_model=192, n_head=6)
modello = AgoGPT(cfg).eval()
B, T = 2, 16
idx = torch.randint(0, cfg.vocab_size, (B, T))

titolo("Che modello e'")
print(f"  {cfg.descrizione()}")
print(f"  parametri: {conta_parametri(modello)}")

titolo("Le forme, dall'inizio alla fine")
stampa_shape("indici in ingresso", idx)
with torch.no_grad():
    emb = modello.tok_emb(idx)
    stampa_shape("dopo l'embedding", emb)
    h = emb
    for i, blocco in enumerate(modello.blocchi):
        h, _ = blocco(h, modello.rope_cos[:T], modello.rope_sin[:T])
        stampa_shape(f"dopo il blocco {i}", h)
    hn = modello.norm_f(h)
    stampa_shape("dopo la norma finale", hn)
    logits = modello.lm_head(hn)
    stampa_shape("logit", logits)
print("  la forma non cambia mai dentro la pila: e' la stessa larghezza dal")
print("  primo blocco all'ultimo, ed e' cio' che rende sensato sommare i")
print("  contributi di blocchi diversi")

titolo("Rifatto a mano")
with torch.no_grad():
    logits_diretti, _ = modello(idx)
verifica(torch.allclose(logits, logits_diretti, atol=1e-5),
         "il percorso ricostruito passo per passo coincide con il forward")

titolo("Appena costruito, non sa niente")
# Il bersaglio e' il token successivo, non il token stesso: con i pesi condivisi
# fra embedding e testa, predire se stessi sarebbe banale anche senza
# addestramento, e la perdita misurata non direbbe nulla.
with torch.no_grad():
    _, perdita = modello(idx[:, :-1], idx[:, 1:])
print(f"  perdita: {perdita.item():.4f}")
print(f"  log(V) = log({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")
verifica(abs(perdita.item() - math.log(cfg.vocab_size)) < 0.4,
         "il modello non addestrato e' vicino alla distribuzione uniforme")

titolo("Una tabella usata due volte")
print(f"  tok_emb.weight e' lm_head.weight: "
      f"{modello.tok_emb.weight is modello.lm_head.weight}")
risparmio = cfg.vocab_size * cfg.d_model
print(f"  parametri risparmiati: {risparmio} su {conta_parametri(modello)} "
      f"({100 * risparmio / (conta_parametri(modello) + risparmio):.1f}%)")
print("  su un vocabolario da 50.000 simboli il risparmio diventa decisivo")

titolo("Non sa nulla del futuro")
idx2 = idx.clone()
idx2[:, -1] = (idx2[:, -1] + 1) % cfg.vocab_size
with torch.no_grad():
    l1, _ = modello(idx)
    l2, _ = modello(idx2)
verifica(torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-6),
         "cambiare l'ultimo token non tocca le previsioni dei precedenti")
verifica(not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-6),
         "ma cambia la previsione della posizione modificata")

titolo("Dove stanno i parametri")
gruppi = {
    "embedding (condivisi)": modello.tok_emb.weight.numel(),
    "attenzione": sum(conta_parametri(b.attn) for b in modello.blocchi),
    "FFN": sum(conta_parametri(b.ffn) for b in modello.blocchi),
    "norme": sum(conta_parametri(b.norm1) + conta_parametri(b.norm2)
                 for b in modello.blocchi) + conta_parametri(modello.norm_f),
}
totale = conta_parametri(modello)
righe = [[k, v, f"{100 * v / totale:.1f}%".replace(".", ",")]
         for k, v in gruppi.items()]
righe.append(["totale", totale, "100,0%"])
tabella(["gruppo", "parametri", "quota"], righe)
print("  la FFN pesa piu' dell'attenzione, con questo rapporto d_ff/d: e' vero")
print("  anche nei modelli grandi, dove le due parti stanno circa due a uno")
