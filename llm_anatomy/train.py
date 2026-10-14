"""Ciclo di addestramento con segmentazione, checkpoint e ripresa esatta.

La ripresa salva anche lo stato dei generatori casuali: e' cio' che rende la
curva di perdita continua attraverso un'interruzione, invece che solo simile.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import AgoGPT
from llm_anatomy.utils import percorso


@dataclass
class ConfigAddestramento:
    passi: int = 2500
    batch_size: int = 16
    lr: float = 3e-3
    lr_min_frazione: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    intervallo_eval: int = 100
    batch_eval: int = 20
    intervallo_checkpoint: int = 100
    seed: int = 1337


def lr_al_passo(passo: int, tc: ConfigAddestramento) -> float:
    """Warmup lineare e poi discesa a coseno fino a lr_min_frazione * lr."""
    if passo < tc.warmup:
        return tc.lr * (passo + 1) / tc.warmup
    avanzamento = (passo - tc.warmup) / max(1, tc.passi - tc.warmup)
    avanzamento = min(1.0, avanzamento)
    coseno = 0.5 * (1 + math.cos(math.pi * avanzamento))
    return tc.lr * (tc.lr_min_frazione + (1 - tc.lr_min_frazione) * coseno)


@torch.no_grad()
def stima_perdita(modello, dati, tc: ConfigAddestramento, generatore) -> float:
    """Perdita media su batch_eval batch estratti con un generatore separato.

    Il generatore e' separato da quello dell'addestramento perche' altrimenti
    valutare consumerebbe numeri casuali e cambierebbe i batch successivi.
    """
    from llm_anatomy.data import get_batch

    modello.eval()
    totale = 0.0
    for _ in range(tc.batch_eval):
        x, y = get_batch(dati, tc.batch_size, modello.cfg.block_size, generatore)
        _, perdita = modello(x, y)
        totale += perdita.item()
    modello.train()
    return totale / tc.batch_eval


def salva_checkpoint(percorso_file: Path, modello, ottimizzatore, passo: int,
                     cfg: AgoGPTConfig, tc: ConfigAddestramento,
                     gen_train, gen_eval) -> None:
    torch.save(
        {
            "modello": modello.state_dict(),
            "ottimizzatore": ottimizzatore.state_dict(),
            "passo": passo,
            "cfg": cfg.as_dict(),
            "tc": asdict(tc),
            "rng_torch": torch.get_rng_state(),
            "rng_train": gen_train.get_state(),
            "rng_eval": gen_eval.get_state(),
        },
        percorso_file,
    )


def carica_checkpoint(percorso_file: Path, device="cpu"):
    """Legge un checkpoint senza eseguire niente di cio' che contiene.

    weights_only=True accetta tensori, numeri e dizionari, e rifiuta tutto il
    resto: un file di pesi scaricato dalla rete non puo' far girare codice.
    """
    return torch.load(percorso_file, map_location=device, weights_only=True)


def addestra(modello: AgoGPT, dati_train, dati_val, tc: ConfigAddestramento,
             nome: str = "agogpt", max_seconds: float | None = None,
             riprendi: bool = True, verboso: bool = True) -> dict:
    """Addestra fino a tc.passi, fermandosi dopo max_seconds secondi.

    Restituisce un riepilogo. Chiamata di nuovo con lo stesso nome, riparte dal
    checkpoint: e' il modo in cui un addestramento sopravvive a una sessione
    che scade o a un portatile che si chiude.
    """
    from llm_anatomy.data import get_batch

    file_ckpt = percorso("checkpoints", f"{nome}.pt")
    file_log = percorso("outputs", f"{nome}_log.jsonl")

    ottimizzatore = torch.optim.AdamW(
        modello.parameters(), lr=tc.lr, betas=(0.9, 0.95),
        weight_decay=tc.weight_decay,
    )
    gen_train = torch.Generator().manual_seed(tc.seed)
    gen_eval = torch.Generator().manual_seed(tc.seed + 1)
    passo = 0

    stato = carica_checkpoint(file_ckpt) if riprendi and file_ckpt.exists() else None
    # Un file con i soli pesi -- quello che si scarica per usare il modello
    # senza addestrarlo -- non porta lo stato dell'ottimizzatore, e da li' non
    # si riprende: si addestra da zero, e alla fine il file viene sostituito.
    if stato is not None and "ottimizzatore" in stato:
        modello.load_state_dict(stato["modello"])
        ottimizzatore.load_state_dict(stato["ottimizzatore"])
        passo = stato["passo"]
        torch.set_rng_state(stato["rng_torch"])
        gen_train.set_state(stato["rng_train"])
        gen_eval.set_state(stato["rng_eval"])
        if verboso:
            print(f"  [ripresa] checkpoint al passo {passo}")

    modello.train()
    t0 = time.perf_counter()
    passi_fatti = 0
    ultima_val = float("nan")

    while passo < tc.passi:
        lr = lr_al_passo(passo, tc)
        for gruppo in ottimizzatore.param_groups:
            gruppo["lr"] = lr

        x, y = get_batch(dati_train, tc.batch_size, modello.cfg.block_size,
                         gen_train)
        _, perdita = modello(x, y)
        ottimizzatore.zero_grad(set_to_none=True)
        perdita.backward()
        torch.nn.utils.clip_grad_norm_(modello.parameters(), tc.grad_clip)
        ottimizzatore.step()
        passo += 1
        passi_fatti += 1

        if passo % tc.intervallo_eval == 0 or passo == tc.passi:
            ultima_val = stima_perdita(modello, dati_val, tc, gen_eval)
            trascorso = time.perf_counter() - t0
            riga = {
                "passo": passo,
                "loss_train": round(perdita.item(), 4),
                "loss_val": round(ultima_val, 4),
                "lr": round(lr, 6),
                "ms_per_passo": round(1000 * trascorso / max(1, passi_fatti), 1),
            }
            with open(file_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(riga) + "\n")
            if verboso:
                print(f"  passo {passo:5d}  train {perdita.item():.3f}  "
                      f"val {ultima_val:.3f}  lr {lr:.2e}  "
                      f"{riga['ms_per_passo']:.0f} ms/passo")

        if passo % tc.intervallo_checkpoint == 0:
            salva_checkpoint(file_ckpt, modello, ottimizzatore, passo,
                             modello.cfg, tc, gen_train, gen_eval)

        if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
            salva_checkpoint(file_ckpt, modello, ottimizzatore, passo,
                             modello.cfg, tc, gen_train, gen_eval)
            if verboso:
                print(f"  [segmento concluso] passo {passo}/{tc.passi}, "
                      f"{time.perf_counter() - t0:.0f} s: checkpoint salvato")
            break

    if passi_fatti == 0:
        # Ripresa di un addestramento gia' concluso: nessun passo da fare, ma
        # la perdita va comunque misurata, altrimenti il chiamante riceverebbe
        # un valore non definito.
        ultima_val = stima_perdita(modello, dati_val, tc, gen_eval)

    salva_checkpoint(file_ckpt, modello, ottimizzatore, passo, modello.cfg, tc,
                     gen_train, gen_eval)
    return {
        "passo": passo,
        "completato": passo >= tc.passi,
        "loss_val": ultima_val,
        "secondi": time.perf_counter() - t0,
        "checkpoint": str(file_ckpt),
    }
