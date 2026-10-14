"""Quanto grande fare ogni passo, e perche' non sempre uguale.

Il tasso di apprendimento decide di quanto si sposta il modello a ogni
correzione. Tenerlo fisso funziona male in due momenti opposti: all'inizio,
quando l'ottimizzatore non ha ancora idea di quanto siano grandi i gradienti,
e alla fine, quando servono aggiustamenti fini e non salti.

Si esegue con `python3 scripts/05_04_schedule_lr.py`
e dura circa 35 secondi.

Da provare: porta il warmup a zero e confronta la perdita a venti passi.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.data import get_batch, prepara_split
from llm_anatomy.model import AgoGPT
from llm_anatomy.train import ConfigAddestramento, lr_al_passo
from llm_anatomy.utils import (dec, figura, prepara, stile_figure, tabella,
                               titolo, verifica)

prepara()

PASSI = 600     # un addestramento corto, per poterlo rifare tre volte
WARMUP = 60     # i primi dieci per cento salgono da zero
LR = 3e-3       # il valore di crociera


# Sale in linea retta per WARMUP passi, poi scende come un quarto di coseno
# fino a un decimo del valore di crociera. E' la forma usata quasi ovunque.
def coseno(passo: int) -> float:
    if passo < WARMUP:
        return LR * (passo + 1) / WARMUP
    avanzamento = (passo - WARMUP) / (PASSI - WARMUP)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * avanzamento)))


def radice_inversa(passo: int, d_model: int = 512) -> float:
    """La formula del paper 2017: lrate = d^-0.5 * min(s^-0.5, s * warmup^-1.5)."""
    s = passo + 1
    return d_model ** -0.5 * min(s ** -0.5, s * WARMUP ** -1.5)


def costante(passo: int) -> float:
    return LR


titolo("Tre modi di scegliere il passo")
righe = []
for passo in (0, 10, WARMUP - 1, WARMUP, 200, 400, PASSI - 1):
    righe.append([passo,
                  f"{coseno(passo):.2e}",
                  f"{radice_inversa(passo):.2e}",
                  f"{costante(passo):.2e}"])
tabella(["passo", "coseno + warmup", "radice inversa (2017)", "costante"], righe)
print("  la radice inversa e' scritta con la normalizzazione del paper, quindi")
print("  su una scala diversa: cio' che conta e' la forma, non il valore")

titolo("Perche' non si parte a tutta velocita'")
print("  ad Adam servono alcuni passi per stimare i momenti del gradiente. Nei")
print("  primissimi passi la stima e' rumorosa, e un tasso pieno la trasforma")
print("  in aggiornamenti enormi. Il warmup non e' scaramanzia: e' il tempo")
print("  che si concede all'ottimizzatore per misurare la scala dei gradienti")
verifica(coseno(0) < coseno(WARMUP) / 10,
         "il tasso al primo passo e' meno di un decimo di quello a regime")
verifica(abs(coseno(PASSI - 1) - 0.1 * LR) < 0.02 * LR,
         "alla fine il coseno si e' ridotto al 10% del valore massimo, come previsto")

titolo("La differenza, misurata")
_, dati_train, dati_val = prepara_split()
cfg = AgoGPTConfig(vocab_size=93, block_size=64, n_layer=2, d_model=96, n_head=4)
tc = ConfigAddestramento(passi=PASSI, batch_size=16, lr=LR, warmup=WARMUP)

risultati = {}
for nome, schedule in (("coseno + warmup", coseno),
                       ("costante", costante),
                       ("costante senza warmup, 3x", lambda p: 3 * LR)):
    torch.manual_seed(1337)
    modello = AgoGPT(cfg)
    ott = torch.optim.AdamW(modello.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(1337)
    storia = []
    for passo in range(PASSI):
        for gruppo in ott.param_groups:
            gruppo["lr"] = schedule(passo)
        x, y = get_batch(dati_train, tc.batch_size, cfg.block_size, gen)
        _, perdita = modello(x, y)
        ott.zero_grad(set_to_none=True)
        perdita.backward()
        torch.nn.utils.clip_grad_norm_(modello.parameters(), 1.0)
        ott.step()
        if (passo + 1) % 20 == 0:
            storia.append(perdita.item())
    gen_eval = torch.Generator().manual_seed(4242)
    with torch.no_grad():
        perdite = []
        for _ in range(20):
            x, y = get_batch(dati_val, tc.batch_size, cfg.block_size, gen_eval)
            _, p = modello(x, y)
            perdite.append(float(p))
    risultati[nome] = (storia, sum(perdite) / len(perdite))

righe = [[nome, dec(storia[0], 3), dec(storia[-1], 3), dec(val, 4)]
         for nome, (storia, val) in risultati.items()]
tabella(["schedule", "perdita a 20 passi", "a fine corsa", "validazione"], righe)

migliore = min(risultati, key=lambda k: risultati[k][1])
print(f"  il migliore e' '{migliore}'")
verifica(risultati["coseno + warmup"][1] <= risultati["costante"][1],
         "il coseno con warmup non fa peggio del tasso costante")
divario = risultati["costante"][1] - risultati["coseno + warmup"][1]
print(f"  il divario sulla validazione e' {dec(divario, 4)} nat: piccolo in")
print("  valore assoluto, e nella stessa direzione a ogni scala. La coda a")
print("  tasso basso serve a rifinire, la fase a tasso alto a non fermarsi nel")
print("  primo minimo incontrato, e il warmup a non rovinare i primi passi:")
print(f"  il tasso triplo senza warmup chiude a {dec(risultati['costante senza warmup, 3x'][1], 4)}")

fig, ax = plt.subplots(2, 1, figsize=(stile_figure(), 4.6))
passi = range(PASSI)
ax[0].plot(passi, [coseno(p) for p in passi], lw=1.4, label="coseno + warmup")
ax[0].plot(passi, [costante(p) for p in passi], lw=1.2, ls="--", label="costante")
ax[0].axvline(WARMUP, ls=":", lw=0.8, color="gray")
ax[0].set_xlabel("passo")
ax[0].set_ylabel("tasso di apprendimento")
ax[0].legend()
ax2 = ax[0].twinx()
ax2.plot(passi, [radice_inversa(p) for p in passi], lw=1.0, color="#6B7076")
ax2.set_ylabel("radice inversa (scala propria)", color="#6B7076")
for nome, (storia, _) in risultati.items():
    ax[1].plot([20 * (i + 1) for i in range(len(storia))], storia, lw=1.2,
               label=nome)
ax[1].set_xlabel("passo")
ax[1].set_ylabel("perdita di addestramento")
ax[1].legend()
figura(fig, "05-04_schedule_lr")
