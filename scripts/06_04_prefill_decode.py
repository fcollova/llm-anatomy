"""Le due fasi di una risposta, che costano in modo opposto.

Leggere la domanda e scrivere la risposta sono due lavori diversi. Il primo
elabora tutto insieme e satura il processore; il secondo procede un carattere
alla volta e passa il tempo ad aspettare la memoria. Quasi tutto quello che si
legge sui costi dei modelli discende da qui.

Si esegue con `python3 scripts/06_04_prefill_decode.py`
e dura circa 10 secondi.

Da provare: porta il batch a 32 e guarda quale delle due fasi migliora.
"""

import time

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import tokenizer_corpus
from llm_anatomy.model import AgoGPT
from llm_anatomy.train import carica_checkpoint
from llm_anatomy.utils import dec, numero, percorso, prepara, tabella, titolo, verifica

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
parametri = modello.numero_parametri()


def mediana(fn, ripetizioni: int = 9) -> float:
    with torch.no_grad():
        for _ in range(2):
            fn()
        tempi = []
        for _ in range(ripetizioni):
            t0 = time.perf_counter()
            fn()
            tempi.append(time.perf_counter() - t0)
    tempi.sort()
    return tempi[len(tempi) // 2]


titolo("Leggere la domanda, scrivere la risposta")
print("  prefill: il prompt entra tutto insieme, ogni posizione e'")
print("           indipendente dalle altre nel calcolo delle proiezioni, e la")
print("           macchina lavora su matrici grandi")
print("  decode:  un token alla volta, matrici a una riga, e gli stessi pesi")
print("           riletti dalla memoria a ogni passo")

titolo("Quanto rendono, a confronto")
righe = []
for lunghezza in (16, 64, 128):
    x = torch.randint(0, cfg.vocab_size, (1, lunghezza))
    t_prefill = mediana(lambda: modello(x))
    caches = modello.nuove_cache()
    with torch.no_grad():
        modello(x, caches=caches)
    uno = torch.randint(0, cfg.vocab_size, (1, 1))
    t_decode = mediana(lambda: modello(uno, caches=modello.nuove_cache()))
    righe.append([
        lunghezza,
        dec(1000 * t_prefill, 2),
        f"{lunghezza / t_prefill:,.0f}".replace(",", "."),
        dec(1000 * t_decode, 2),
        f"{1 / t_decode:,.0f}".replace(",", "."),
    ])
tabella(["prompt", "prefill (ms)", "token/s prefill", "decode (ms)",
         "token/s decode"], righe)

x = torch.randint(0, cfg.vocab_size, (1, 128))
uno = torch.randint(0, cfg.vocab_size, (1, 1))
t_prefill = mediana(lambda: modello(x))
t_decode = mediana(lambda: modello(uno, caches=modello.nuove_cache()))
print(f"  un token in prefill costa {dec(1000 * t_prefill / 128, 3)} ms,")
print(f"  un token in decode {dec(1000 * t_decode, 3)} ms: "
      f"{dec(t_decode / (t_prefill / 128), 1)} volte tanto")
verifica(t_decode > 5 * (t_prefill / 128),
         "un token generato costa molte volte un token letto")

titolo("Le due attese che si notano")
prompt = 128
for da_generare in (20, 100):
    ttft = t_prefill
    tpot = t_decode
    totale = ttft + da_generare * tpot
    print(f"  prompt {prompt} token, generazione {da_generare} token:")
    print(f"    tempo al primo token (TTFT): {dec(1000 * ttft, 1)} ms")
    print(f"    tempo fra token (TPOT):      {dec(1000 * tpot, 1)} ms")
    print(f"    totale:                      {dec(totale, 3)} s, di cui "
          f"{dec(100 * ttft / totale, 1)}% di prefill")
print("  con prompt lunghi e risposte brevi domina il prefill, con prompt")
print("  brevi e risposte lunghe domina il decode: sono due problemi diversi")
print("  e si ottimizzano con tecniche diverse")

titolo("Servire piu' utenti aiuta una fase sola")
righe = []
for B in (1, 8, 32):
    xb = torch.randint(0, cfg.vocab_size, (B, 128))
    t_p = mediana(lambda: modello(xb), ripetizioni=5)
    unob = torch.randint(0, cfg.vocab_size, (B, 1))
    t_d = mediana(lambda: modello(unob, caches=modello.nuove_cache()))
    righe.append([B,
                  dec(1000 * t_p, 1),
                  numero(B * 128 / t_p),
                  dec(1000 * t_d, 2),
                  numero(B / t_d)])
tabella(["batch", "prefill (ms)", "token/s prefill", "decode (ms)",
         "token/s decode"], righe)
print("  il prefill era gia' saturo con una sola sequenza: il batch lo")
print("  rallenta quasi proporzionalmente. Il decode invece era in attesa")
print("  della memoria, e servire piu' sequenze insieme costa quasi lo stesso")
print("  tempo a passo: e' su questo che si regge il batching continuo dei")
print("  motori di servizio")

titolo("Byte letti per ogni carattere scritto")
byte = parametri * 4
print(f"  parametri: {parametri:,}".replace(",", "."))
print(f"  byte letti a ogni passo di decode: {byte / 1e6:.2f} MB"
      .replace(".", ","))
print(f"  banda effettiva in decode: {byte / t_decode / 1e9:.2f} GB/s"
      .replace(".", ","))
print("  e' il conto che spiega perche' il decode di un modello grande e'")
print("  lento anche su hardware potente: si legge tutto il modello per")
print("  produrre un solo token")
