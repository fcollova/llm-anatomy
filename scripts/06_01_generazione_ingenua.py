"""Generare nel modo ovvio, e scoprire che non va.

Per scrivere il carattere successivo si ridanno in pasto al modello tutti i
caratteri scritti finora. Funziona, ed e' quello che verrebbe in mente a
chiunque. Il problema e' che il costo cresce con il quadrato della lunghezza,
e si vede gia' su cento caratteri.

Si esegue con `python3 scripts/06_01_generazione_ingenua.py`
e dura circa 10 secondi.

Da provare: genera 200 caratteri invece di 100 e guarda il tempo quadruplicare.
"""

import time

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import tokenizer_corpus
from llm_anatomy.model import AgoGPT
from llm_anatomy.train import carica_checkpoint
from llm_anatomy.utils import dec, percorso, prepara, tabella, titolo, verifica

prepara()

# Questo script misura tempi, e con otto thread il rumore di scheduling supera
# il segnale su un modello cosi' piccolo. Per le sole misure si torna a un
# thread: e' la condizione in cui i tempi sono confrontabili, ed e' cio' che
# chiedono le convenzioni di redazione.
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

titolo("Il ciclo che verrebbe in mente a chiunque")
print("  per ogni token da generare:")
print("    1. si passa l'intero prefisso nel modello")
print("    2. si prendono i logit dell'ultima posizione")
print("    3. si campiona e si appende")
print("  il passo 1 rifa' ogni volta il lavoro gia' fatto")

avvio = torch.tensor([tok.encode("Risotto coi funghi\n\n")])
N_TOKEN = 100

titolo("Ogni carattere costa piu' del precedente")
# Una sola misura per lunghezza sarebbe rumore: il sistema operativo, la
# cache e gli altri processi spostano i tempi di frazioni di millisecondo, che
# a questa scala sono una parte grossa del totale. Per ogni lunghezza si
# cronometrano allora sette passaggi e si tiene la mediana.
def cronometra(lunghezza: int, ripetizioni: int = 15) -> float:
    x = torch.randint(0, cfg.vocab_size, (4, lunghezza))
    with torch.no_grad():
        for _ in range(2):
            modello(x)
        tempi_l = []
        for _ in range(ripetizioni):
            t0 = time.perf_counter()
            modello(x)
            tempi_l.append(time.perf_counter() - t0)
    tempi_l.sort()
    return tempi_l[len(tempi_l) // 2]


lunghezze = list(range(8, cfg.block_size + 1, 8))
tempi = [cronometra(l) for l in lunghezze]

righe = []
for l, tt in zip(lunghezze, tempi):
    if l in (8, 32, 64, 96, 128):
        righe.append([l, dec(1000 * tt, 3), dec(1000 * tt / l, 4)])
tabella(["lunghezza del prefisso", "ms per passaggio", "ms per posizione"], righe)

print(f"  il passaggio su {lunghezze[-1]} token costa "
      f"{dec(tempi[-1] / tempi[0], 2)} volte quello su {lunghezze[0]}")
verifica(tempi[-1] > tempi[0],
         "un prefisso piu' lungo costa piu' tempo, a parita' di token prodotto")

titolo("Quanto cresce, esattamente")
import numpy as np

xs = np.array(lunghezze, dtype=float)
ms = np.array(tempi) * 1000
pendenza, intercetta = np.polyfit(xs, ms, 1)
predette = pendenza * xs + intercetta
r2 = 1 - float(np.sum((ms - predette) ** 2) / np.sum((ms - ms.mean()) ** 2))
print(f"  costo fisso per chiamata:    {dec(float(intercetta), 3)} ms")
print(f"  costo per token di prefisso: {dec(float(pendenza), 4)} ms")
print(f"  bonta' del fit (R^2):        {dec(r2, 3)}")
verifica(pendenza > 0 and r2 > 0.8,
         "il costo di un passaggio cresce linearmente con la lunghezza")
print("  generare n token senza cache significa fare n passaggi di lunghezza")
print("  crescente: la somma di una retta e' una parabola, ed e' da li' che")
print("  viene il costo quadratico")

righe = []
for n in (128, 1024, 8192):
    totale_ms = n * intercetta + pendenza * n * (n - 1) / 2
    righe.append([n, dec(totale_ms / 1000, 2), dec(totale_ms / n, 2)])
tabella(["token generati", "secondi (estrapolati)", "ms per token"], righe)
print("  la tabella estrapola la retta appena misurata: questo modello ha un")
print("  contesto di 128 e oltre quella soglia tronca, ma su un modello a")
print("  contesto lungo e' esattamente cio' che si paga")

titolo("Quanto lavoro viene rifatto ogni volta")
n = 100
utili = n
ripetute = sum(range(n))
print(f"  generando {n} token, il modello elabora "
      f"{utili + ripetute} posizioni in totale")
print(f"  ne servirebbero {utili}: le altre {ripetute} sono ricalcoli identici")
print(f"  lavoro sprecato: {dec(100 * ripetute / (utili + ripetute), 1)}%")
print("  ogni chiave e ogni valore gia' calcolati vengono ricalcolati identici")
print("  a ogni passo, perche' dipendono solo dal token e dalla sua posizione")

titolo("Perche' si possono conservare")
byte_per_token = 2 * cfg.n_layer * cfg.n_kv_head * cfg.d_head * 4
print(f"  memoria per conservarli: {byte_per_token} byte per token")
print(f"  per una sequenza da {cfg.block_size}: "
      f"{byte_per_token * cfg.block_size / 1024:.1f} KiB"
      .replace(".", ","))
print("  e' il baratto che lo script 06-02 misura: memoria contro tempo")
