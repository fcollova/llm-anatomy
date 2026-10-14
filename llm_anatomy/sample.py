"""Campionatori e generazione, con e senza cache.

Le funzioni prendono e restituiscono logit: sono pure, quindi lo script 06-03
puo' applicarle tutte agli stessi logit e mostrare che la differenza fra i
campionatori sta solo in quali code della distribuzione vengono tagliate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temperatura(logits: torch.Tensor, t: float) -> torch.Tensor:
    """t<1 concentra, t>1 appiattisce. t=0 e' il caso limite: greedy."""
    if t <= 0:
        fuori = torch.full_like(logits, float("-inf"))
        fuori.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
        return fuori
    return logits / t


def top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Tiene i k logit piu' alti. Taglio a numero fisso, indipendente dalla forma."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    soglia = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < soglia, float("-inf"))


def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus: tiene i simboli piu' probabili fino a coprire massa p."""
    if p >= 1.0:
        return logits
    ordinati, indici = logits.sort(dim=-1, descending=True)
    cumulata = ordinati.softmax(dim=-1).cumsum(dim=-1)
    da_togliere = cumulata - ordinati.softmax(dim=-1) >= p
    ordinati = ordinati.masked_fill(da_togliere, float("-inf"))
    return ordinati.gather(-1, indici.argsort(dim=-1))


def min_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Soglia relativa al massimo: taglia sotto p * probabilita' del migliore.

    A differenza di top-p il taglio si adatta da solo: se il modello e' sicuro
    resta quasi solo il vincitore, se e' incerto la finestra si allarga.
    """
    if p <= 0:
        return logits
    prob = logits.softmax(dim=-1)
    soglia = p * prob.max(dim=-1, keepdim=True).values
    return logits.masked_fill(prob < soglia, float("-inf"))


def campiona(logits: torch.Tensor, temp: float = 1.0, k: int = 0,
             p: float = 1.0, minp: float = 0.0,
             generatore: torch.Generator | None = None) -> torch.Tensor:
    """Applica i filtri nell'ordine consueto e estrae un simbolo."""
    z = temperatura(logits, temp)
    if k:
        z = top_k(z, k)
    if p < 1.0:
        z = top_p(z, p)
    if minp:
        z = min_p(z, minp)
    prob = z.softmax(dim=-1)
    return torch.multinomial(prob, num_samples=1, generator=generatore)


@torch.no_grad()
def genera(modello, idx: torch.Tensor, n_token: int, temp: float = 1.0,
           k: int = 0, p: float = 1.0, minp: float = 0.0,
           usa_cache: bool = True,
           generatore: torch.Generator | None = None) -> torch.Tensor:
    """Genera n_token continuando idx (B, T).

    Con usa_cache=False ogni passo ricalcola l'intero prefisso: e' la versione
    ingenua dello script 06-01, tenuta qui per poter misurare la differenza
    senza riscrivere due volte il ciclo di generazione.
    """
    modello.eval()
    T_max = modello.cfg.block_size
    caches = modello.nuove_cache() if usa_cache else None
    offset = 0
    ingresso = idx

    for _ in range(n_token):
        if usa_cache:
            if offset >= T_max:
                # Finestra piena: la cache non e' piu' utilizzabile, perche' le
                # posizioni oltre block_size non esistono. Si ricostruisce sugli
                # ultimi T_max token, che e' esattamente cio' che farebbe la
                # versione senza cache: da qui in poi i due percorsi restano
                # identici, e il vantaggio della cache sparisce.
                caches = modello.nuove_cache()
                ingresso = idx[:, -T_max:]
                offset = 0
            else:
                ingresso = (ingresso[:, -T_max:] if offset == 0
                            else ingresso[:, -1:])
            logits, _ = modello(ingresso, caches=caches, pos_offset=offset)
            offset += ingresso.shape[1]
        else:
            logits, _ = modello(idx[:, -T_max:])

        prossimo = campiona(logits[:, -1, :], temp, k, p, minp, generatore)
        idx = torch.cat([idx, prossimo], dim=1)
        ingresso = idx
    return idx
