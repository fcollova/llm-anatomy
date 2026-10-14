"""Conservare quello che e' gia' stato calcolato.

Il lavoro rifatto a ogni carattere non cambia mai, perche' la maschera causale
impedisce al passato di dipendere dal futuro. Se non cambia, si puo' tenere da
parte: e' la KV cache, e trasforma un costo quadratico in uno lineare.

Si esegue con `python3 scripts/06_02_kv_cache.py`
e dura circa 10 secondi.

Da provare: misura la memoria della cache a contesto 512.
"""

import time

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import tokenizer_corpus
from llm_anatomy.model import AgoGPT
from llm_anatomy.sample import genera
from llm_anatomy.train import carica_checkpoint
from llm_anatomy.utils import (byte_leggibili, dec, percorso, prepara,
                               tabella, titolo, verifica)

prepara()
torch.set_num_threads(1)
print("[misure] torch.set_num_threads(1) per rendere i tempi confrontabili")

file_ckpt = percorso("checkpoints", "agogpt.pt")
if not file_ckpt.exists():
    raise SystemExit("checkpoint assente: eseguire prima 05-02")
stato = carica_checkpoint(file_ckpt)
cfg = AgoGPTConfig(**stato["cfg"])
modello = AgoGPT(cfg)
modello.load_state_dict(stato["modello"])
modello.eval()
tok = tokenizer_corpus()

avvio = torch.tensor([tok.encode("Risotto coi funghi\n\n")])

titolo("Cio' che non cambia si conserva")
print("  chiavi e valori di un token dipendono solo da quel token e dalla sua")
print("  posizione: una volta calcolati non cambiano piu'. La cache li tiene,")
print("  e a ogni passo il modello elabora un solo token nuovo invece di tutto")
print("  il prefisso")

titolo("Stesso testo, cifra per cifra")
for temperatura, etichetta in ((0.0, "greedy"), (0.8, "con campionamento")):
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    con = genera(modello, avvio.clone(), 60, temp=temperatura, usa_cache=True,
                 generatore=g1)
    senza = genera(modello, avvio.clone(), 60, temp=temperatura, usa_cache=False,
                   generatore=g2)
    uguali = torch.equal(con, senza)
    print(f"  {etichetta:<20} identici: {uguali}")
    verifica(uguali, f"la cache non cambia il risultato ({etichetta})")
print("  con il campionamento la verifica e' piu' forte: gli stessi numeri")
print("  casuali devono cadere sugli stessi logit, altrimenti le due sequenze")
print("  divergerebbero al primo scarto numerico")

titolo("Quanto si guadagna")
def cronometra(n_token: int, cache: bool, ripetizioni: int = 3) -> float:
    tempi = []
    for _ in range(ripetizioni):
        t0 = time.perf_counter()
        genera(modello, avvio.clone(), n_token, temp=0.0, usa_cache=cache)
        tempi.append(time.perf_counter() - t0)
    tempi.sort()
    return tempi[len(tempi) // 2]


righe = []
for n in (16, 32, 64, 100):
    t_con = cronometra(n, True)
    t_senza = cronometra(n, False)
    righe.append([n, dec(t_senza, 3), dec(t_con, 3), dec(t_senza / t_con, 2) + "x",
                  dec(1000 * t_con / n, 2)])
tabella(["token generati", "senza cache (s)", "con cache (s)", "speedup",
         "ms per token con cache"], righe)
guadagni = [float(r[3][:-1].replace(",", ".")) for r in righe]
verifica(guadagni[-1] > guadagni[0],
         "il vantaggio della cache cresce con la lunghezza della sequenza")
print("  lo speedup non e' una costante: cresce, perche' e' il rapporto fra un")
print("  costo quadratico e uno lineare")

titolo("Il prezzo si paga in memoria")
caches = modello.nuove_cache()
idx = avvio.clone()
righe = []
with torch.no_grad():
    offset = 0
    for passo in range(cfg.block_size - avvio.shape[1]):
        ingresso = idx if offset == 0 else idx[:, -1:]
        logits, _ = modello(ingresso, caches=caches, pos_offset=offset)
        offset += ingresso.shape[1]
        idx = torch.cat([idx, logits[:, -1:].argmax(dim=-1)], dim=1)
        if offset in (18, 32, 64, 96, 128):
            byte = sum(c.byte() for c in caches)
            righe.append([offset, byte_leggibili(byte),
                          byte_leggibili(byte / offset)])
tabella(["token in cache", "memoria totale", "per token"], righe)

per_token = sum(c.byte() for c in caches) / offset
print(f"  la crescita e' lineare: {byte_leggibili(per_token)} per token, cioe'")
print(f"  2 tensori x {cfg.n_layer} blocchi x {cfg.n_kv_head} teste x "
      f"{cfg.d_head} dimensioni x 4 byte")

titolo("Che cosa significa su un modello vero")
righe = []
for nome, n_layer, kv, d_head, contesto in (
    ("mini-modello, ctx 128", cfg.n_layer, cfg.n_kv_head, cfg.d_head, 128),
    ("7B GQA a 8, ctx 8k", 32, 8, 128, 8192),
    ("7B GQA a 8, ctx 128k", 32, 8, 128, 131072),
):
    byte_token = 2 * n_layer * kv * d_head * (4 if "mini" in nome else 2)
    righe.append([nome, byte_leggibili(byte_token),
                  byte_leggibili(byte_token * contesto)])
tabella(["configurazione", "per token", "a contesto pieno"], righe)
print("  a contesto lungo la cache diventa piu' grande dei pesi stessi: e' il")
print("  motivo per cui esistono GQA, la quantizzazione della cache e le")
print("  finestre scorrevoli. Il problema non e' piu' generare, e' ricordare")
