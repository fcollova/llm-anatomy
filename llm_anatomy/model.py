"""AgoGPT: il modello del libro, scritto per essere letto.

Decoder-only pre-norm con RMSNorm, RoPE, SwiGLU e GQA opzionale, cioe' la
configurazione corrente dei modelli linguistici. Ogni pezzo e' una classe
piccola, perche' gli script la importano isolata per provarla da sola.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_anatomy.config import AgoGPTConfig


# --------------------------------------------------------------------------
# Normalizzazione
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Normalizza per la sola radice della media dei quadrati.

    Rispetto a LayerNorm mancano la sottrazione della media e il termine di
    bias: due operazioni in meno per token e, in pratica, nessuna perdita di
    qualita'. Vedi lo script 03-08 per il confronto numerico.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.peso = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.peso


def crea_norm(cfg: AgoGPTConfig) -> nn.Module:
    if cfg.norm == "rms":
        return RMSNorm(cfg.d_model)
    if cfg.norm == "layer":
        return nn.LayerNorm(cfg.d_model, bias=cfg.bias)
    raise ValueError(f"norm sconosciuta: {cfg.norm}")


# --------------------------------------------------------------------------
# Posizione
# --------------------------------------------------------------------------
def frequenze_rope(d_head: int, block_size: int, theta: float = 10000.0,
                   scaling: float = 1.0, device=None):
    """Precalcola cos e sin per RoPE, forma (T, d_head/2).

    scaling > 1 comprime le posizioni (interpolazione lineare): la posizione m
    viene trattata come m/scaling, che e' il modo piu' semplice di estendere il
    contesto oltre quello visto in addestramento.
    """
    pari = torch.arange(0, d_head, 2, device=device).float()
    inv = 1.0 / (theta ** (pari / d_head))
    pos = torch.arange(block_size, device=device).float() / scaling
    ang = torch.outer(pos, inv)
    return ang.cos(), ang.sin()


def applica_rope(x: torch.Tensor, cos: torch.Tensor,
                 sin: torch.Tensor) -> torch.Tensor:
    """Ruota a coppie le componenti di x. x: (B, h, T, d_head)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    fuso = torch.stack((r1, r2), dim=-1)
    return fuso.flatten(-2)


def posizionali_sinusoidali(block_size: int, d_model: int) -> torch.Tensor:
    """Tabella (T, d) del paper 2017: seni e coseni a frequenze geometriche."""
    pos = torch.arange(block_size).float().unsqueeze(1)
    i = torch.arange(0, d_model, 2).float()
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(block_size, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# --------------------------------------------------------------------------
# Attenzione
# --------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Attenzione causale multi-testa, con GQA e cache KV opzionali.

    Le proiezioni sono tre matrici separate invece di una fusa: costa un poco
    in prestazioni e restituisce in cambio la possibilita' di stampare Q, K e V
    separatamente, che e' il punto degli script del capitolo 3.
    """

    def __init__(self, cfg: AgoGPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.d_head = cfg.d_head
        self.ripetizioni = cfg.n_head // cfg.n_kv_head
        self.usa_rope = cfg.pos == "rope"
        self.dropout = cfg.dropout

        d_q = cfg.n_head * cfg.d_head       # tutte le teste delle domande
        d_kv = cfg.n_kv_head * cfg.d_head   # quelle di chiavi e valori
        self.wq = nn.Linear(cfg.d_model, d_q, bias=cfg.bias)
        self.wk = nn.Linear(cfg.d_model, d_kv, bias=cfg.bias)
        self.wv = nn.Linear(cfg.d_model, d_kv, bias=cfg.bias)
        self.wo = nn.Linear(d_q, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos=None, sin=None, cache=None,
                restituisci_pesi: bool = False):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)

        if self.usa_rope and cos is not None:
            q = applica_rope(q, cos, sin)
            k = applica_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.aggiorna(k, v)

        # GQA: ogni gruppo di teste Q condivide una testa K/V, replicata qui.
        if self.ripetizioni > 1:
            k = k.repeat_interleave(self.ripetizioni, dim=1)
            v = v.repeat_interleave(self.ripetizioni, dim=1)

        Tk = k.shape[2]
        pesi = None
        if restituisci_pesi:
            punteggi = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            maschera = torch.ones(T, Tk, dtype=torch.bool,
                                  device=x.device).tril(Tk - T)
            punteggi = punteggi.masked_fill(~maschera, float("-inf"))
            pesi = punteggi.softmax(dim=-1)
            y = pesi @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=(T > 1),
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, self.n_head * self.d_head)
        y = self.drop(self.wo(y))
        return (y, pesi) if restituisci_pesi else (y, None)


class KVCache:
    """Cache delle chiavi e dei valori di un blocco, concatenata nel tempo."""

    def __init__(self):
        self.k = None
        self.v = None

    def aggiorna(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def lunghezza(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def byte(self) -> int:
        if self.k is None:
            return 0
        return (self.k.numel() + self.v.numel()) * self.k.element_size()


# --------------------------------------------------------------------------
# Rete feed-forward
# --------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """FFN con porta: due proiezioni in salita, una moltiplicazione, una in discesa."""

    def __init__(self, cfg: AgoGPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class FFN(nn.Module):
    """FFN classica del paper: salita, non linearita', discesa."""

    def __init__(self, cfg: AgoGPTConfig):
        super().__init__()
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.att = F.gelu if cfg.act == "gelu" else F.relu
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(self.att(self.w_up(x))))


def crea_ffn(cfg: AgoGPTConfig) -> nn.Module:
    return SwiGLU(cfg) if cfg.act == "swiglu" else FFN(cfg)


# --------------------------------------------------------------------------
# Blocco e modello
# --------------------------------------------------------------------------
class Blk(nn.Module):
    """Un blocco: attenzione e FFN, ciascuna con la sua norma e il suo residuo.

    pre_norm=False riproduce la disposizione del paper 2017 (norma dopo il
    residuo). Serve allo script 03-08 per mostrare cosa succede ai gradienti.
    """

    def __init__(self, cfg: AgoGPTConfig, pre_norm: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.norm1 = crea_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = crea_norm(cfg)
        self.ffn = crea_ffn(cfg)

    def forward(self, x, cos=None, sin=None, cache=None,
                restituisci_pesi=False):
        if self.pre_norm:
            a, pesi = self.attn(self.norm1(x), cos, sin, cache,
                                restituisci_pesi)
            x = x + a
            x = x + self.ffn(self.norm2(x))
        else:
            a, pesi = self.attn(x, cos, sin, cache, restituisci_pesi)
            x = self.norm1(x + a)
            x = self.norm2(x + self.ffn(x))
        return x, pesi


class AgoGPT(nn.Module):
    """Il modello completo: embedding, pila di blocchi, norma finale, testa."""

    def __init__(self, cfg: AgoGPTConfig, pre_norm: bool = True):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        if cfg.pos == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        elif cfg.pos == "sinusoidal":
            self.register_buffer("pos_tab", posizionali_sinusoidali(
                cfg.block_size, cfg.d_model))

        if cfg.pos == "rope":
            cos, sin = frequenze_rope(cfg.d_head, cfg.block_size,
                                      cfg.rope_theta, cfg.rope_scaling)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.blocchi = nn.ModuleList(
            [Blk(cfg, pre_norm) for _ in range(cfg.n_layer)])
        self.norm_f = crea_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_pesi)
        # Inizializzazione scalata sulla profondita' per le proiezioni di uscita:
        # senza, la varianza del residual stream cresce con il numero di blocchi.
        for nome, p in self.named_parameters():
            if nome.endswith("wo.weight") or nome.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_pesi(modulo):
        if isinstance(modulo, nn.Linear):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
            if modulo.bias is not None:
                nn.init.zeros_(modulo.bias)
        elif isinstance(modulo, nn.Embedding):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None, pos_offset: int = 0,
                restituisci_pesi: bool = False,
                restituisci_stati: bool = False):
        B, T = idx.shape
        x = self.tok_emb(idx)

        cos = sin = None
        if self.cfg.pos == "rope":
            cos = self.rope_cos[pos_offset:pos_offset + T]
            sin = self.rope_sin[pos_offset:pos_offset + T]
        elif self.cfg.pos == "learned":
            posizioni = torch.arange(pos_offset, pos_offset + T,
                                     device=idx.device)
            x = x + self.pos_emb(posizioni)
        elif self.cfg.pos == "sinusoidal":
            x = x + self.pos_tab[pos_offset:pos_offset + T]

        x = self.drop(x)
        stati, pesi_per_blocco = [x], []
        for i, blocco in enumerate(self.blocchi):
            cache = caches[i] if caches is not None else None
            x, pesi = blocco(x, cos, sin, cache, restituisci_pesi)
            stati.append(x)
            if restituisci_pesi:
                pesi_per_blocco.append(pesi)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        perdita = None
        if targets is not None:
            perdita = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )

        if restituisci_pesi or restituisci_stati:
            extra = {}
            if restituisci_pesi:
                extra["pesi"] = pesi_per_blocco
            if restituisci_stati:
                extra["stati"] = stati
            return logits, perdita, extra
        return logits, perdita

    def nuove_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocchi]

    def numero_parametri(self, senza_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if senza_embedding:
            n -= self.tok_emb.weight.numel()
            if self.cfg.pos == "learned":
                n -= self.pos_emb.weight.numel()
        return n
