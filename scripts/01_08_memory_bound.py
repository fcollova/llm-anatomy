"""Perche' generare e' lento anche quando il calcolo e' poco.

Un modello che scrive un carattere alla volta passa la maggior parte del tempo
ad aspettare che i suoi pesi arrivino dalla memoria, non a moltiplicare. E' la
cosa meno intuitiva del libro e si misura in un minuto: la stessa
moltiplicazione rende dieci volte tanto se la si fa su piu' righe insieme.

Si esegue con `python3 scripts/01_08_memory_bound.py`
e dura circa 10 secondi.

Da provare: cambia la taglia della matrice e rifai il conto dei byte per operazione.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import AgoGPT
from llm_anatomy.sample import genera
from llm_anatomy.utils import (byte_leggibili, dec, figura, numero, prepara, stile_figure,
                               tabella, titolo)

prepara()


def cronometra(fn, ripetizioni: int, riscaldamento: int = 3) -> float:
    """Tempo mediano per chiamata, in secondi."""
    for _ in range(riscaldamento):
        fn()
    tempi = []
    for _ in range(ripetizioni):
        t0 = time.perf_counter()
        fn()
        tempi.append(time.perf_counter() - t0)
    tempi.sort()
    return tempi[len(tempi) // 2]


titolo("La stessa moltiplicazione, su piu' righe")
d = 1024
W = torch.randn(d, d)
byte_pesi = W.numel() * W.element_size()
print(f"  matrice dei pesi: {d}x{d}, {byte_leggibili(byte_pesi)}")
print()

righe = []
misure = []            # (batch, FLOP per byte, GFLOP/s), per la figura
for B in (1, 4, 16, 64, 256, 1024):
    X = torch.randn(B, d)
    t = cronometra(lambda: X @ W, ripetizioni=20 if B < 256 else 8)
    flop = 2 * B * d * d
    byte = byte_pesi + X.numel() * 4 + B * d * 4
    misure.append((B, flop / byte, flop / t / 1e9))
    righe.append([
        B,
        dec(1000 * t, 3),
        dec(flop / t / 1e9, 1),
        dec(byte / t / 1e9, 1),
        dec(flop / byte, 1),
    ])
tabella(["batch", "ms", "GFLOP/s", "GB/s letti", "FLOP/byte"], righe)
print("  a batch 1 il rendimento aritmetico e' minimo e la banda e' satura:")
print("  i pesi vanno letti per intero comunque, e servono una riga sola")

titolo("Lo stesso effetto su un modello vero")
cfg = AgoGPTConfig(vocab_size=93, n_layer=4, d_model=192, n_head=6)
modello = AgoGPT(cfg).eval()
print(f"  {cfg.descrizione()}")
print(f"  parametri: {modello.numero_parametri():,}".replace(",", "."))
print()

idx = torch.randint(0, cfg.vocab_size, (1, 64))
with torch.no_grad():
    t_prefill = cronometra(lambda: modello(idx), ripetizioni=10)
    t_decode = cronometra(lambda: modello(idx[:, -1:]), ripetizioni=20)
print(f"  prefill di 64 token: {1000 * t_prefill:7.2f} ms  "
      f"-> {64 / t_prefill:8.0f} token/s")
print(f"  decode di 1 token:   {1000 * t_decode:7.2f} ms  "
      f"-> {1 / t_decode:8.0f} token/s")
print(f"  un token in prefill costa {dec(t_decode / (t_prefill / 64), 1)} volte meno")
print("  di un token in decode, a parita' di modello e di pesi letti")

titolo("Modelli piu' grandi, generazione piu' lenta")
righe = []
for d_model, n_layer in ((96, 2), (192, 4), (384, 6), (768, 8)):
    cfg_i = AgoGPTConfig(vocab_size=93, n_layer=n_layer, d_model=d_model,
                          n_head=max(2, d_model // 64))
    m = AgoGPT(cfg_i).eval()
    par = m.numero_parametri()
    x = torch.randint(0, cfg_i.vocab_size, (1, 1))
    with torch.no_grad():
        t = cronometra(lambda: m(x), ripetizioni=15)
    byte_letti = par * 4
    righe.append([
        f"L{n_layer} d{d_model}",
        f"{par / 1e6:.2f}".replace(".", ","),
        f"{1000 * t:.2f}".replace(".", ","),
        f"{1 / t:.0f}",
        f"{byte_letti / t / 1e9:.1f}".replace(".", ","),
    ])
tabella(["modello", "M par.", "ms/token", "token/s", "GB/s"], righe)
print("  dal secondo modello in poi il tempo per token cresce quasi come i")
print("  parametri, mentre la banda letta si avvicina lentamente al massimo")
print("  della macchina: e' il costo di leggere i pesi, non di moltiplicarli")

titolo("Servire molti utenti costa quasi come servirne uno")
cfg_b = AgoGPTConfig(vocab_size=93, n_layer=4, d_model=384, n_head=6)
mb = AgoGPT(cfg_b).eval()
righe = []
for B in (1, 8, 32, 128):
    x = torch.randint(0, cfg_b.vocab_size, (B, 1))
    with torch.no_grad():
        t = cronometra(lambda: mb(x), ripetizioni=10)
    righe.append([B, dec(1000 * t), numero(B / t)])
tabella(["batch", "ms/passo", "token/s totali"], righe)
print("  moltiplicare per 128 le sequenze non moltiplica per 128 il tempo:")
print("  e' il motivo per cui servire un modello conviene solo in batch")

titolo("Le stesse misure, disegnate")
# Il tetto non e' un dato di targa: lo si ricava dalle misure appena fatte. Il
# tratto orizzontale e' il meglio che la macchina ha reso in GFLOP/s, quello
# obliquo e' la banda migliore osservata moltiplicata per l'intensita'.
picco_flops = max(g for _, _, g in misure)
# GB/s impliciti: a batch basso il tempo e' tutto lettura, quindi il rapporto
# fra resa e intensita' e' la banda che la macchina ha davvero sostenuto.
banda = max(g / i for _, i, g in misure)
soglia = picco_flops / banda
print(f"  picco osservato:      {picco_flops:.0f} GFLOP/s")
print(f"  banda implicita:      {banda:.1f} GB/s")
print(f"  soglia (ginocchio):   {soglia:.0f} operazioni per byte")
print("  sotto quella soglia il tempo lo detta la memoria, sopra il calcolo.")
print("  Il tetto qui non e' un dato di targa ma l'inviluppo di queste stesse")
print("  misure: due punti lo toccano per costruzione, quello a intensita'")
print("  minima che fissa la banda e quello di resa migliore che fissa il")
print("  picco. Cio' che si osserva e' quanto gli altri restino sotto.")
print()
for B, intensita, gflops in misure:
    tetto = min(picco_flops, banda * intensita)
    print(f"  batch {B:>4}: intensita {intensita:6.1f}  "
          f"resa {gflops:6.1f} GFLOP/s  ({100 * gflops / tetto:3.0f}% del tetto)")
print()
print(f"  una CPU ha un rapporto calcolo/banda basso: la soglia sta a "
      f"{soglia:.0f} FLOP/byte.")
print("  Su un acceleratore da centinaia di TFLOP/s e pochi TB/s la stessa")
print("  soglia sale a qualche centinaio, ed e' il motivo per cui li' il")
print("  decode a un utente e' ancora piu' lontano dal tetto che qui")

fig, ax = plt.subplots(figsize=(stile_figure(), 3.2))
x_tetto = [0.2, soglia, 1000]
y_tetto = [banda * 0.2, picco_flops, picco_flops]
ax.plot(x_tetto, y_tetto, color="#1f4e79", lw=2, label="tetto misurato")
ax.scatter([i for _, i, _ in misure], [g for _, _, g in misure],
           color="#c0504d", zorder=3, label="matmul, batch da 1 a 1024")
# I punti cadono dove li mette la misura, che cambia a ogni esecuzione: una
# posizione fissa per le etichette prima o poi ne sovrappone due. Ognuna sta
# in colonna col suo punto, una sotto e la successiva sopra: due punti vicini
# hanno l'etichetta da parti opposte, e nessuna finisce accanto al punto
# sbagliato.
for k, (B, intensita, gflops) in enumerate(misure):
    sotto = k % 2 == 0
    ax.annotate(f"B={B}", (intensita, gflops), textcoords="offset points",
                xytext=(0, -12) if sotto else (0, 6), ha="center",
                color="#c0504d", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
ax.axvline(soglia, color="gray", ls="--", lw=0.8)
ax.annotate("soglia", (soglia, banda * 0.35), rotation=90,
            color="gray", textcoords="offset points", xytext=(-12, 0))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("intensità aritmetica (FLOP per byte letto)")
ax.set_ylabel("prestazioni (GFLOP/s)")
ax.legend(loc="lower right")
ax.grid(True, which="both", lw=0.3, alpha=0.5)
figura(fig, "01-08_roofline")
print("  i punti restano sotto il tetto e lo raggiungono da sinistra:")
print("  e' la figura che si trova nei manuali, qui con numeri misurati")
