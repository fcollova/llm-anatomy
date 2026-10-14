"""
Il modello del libro, e il filo conduttore di tutti gli script.

Espone il modello (AgoGPT), la sua configurazione, il tokenizer a caratteri,
il caricamento dati, il ciclo di addestramento con ripresa da checkpoint e i
campionatori. Nessuno script reimplementa queste parti: le importa, cosi' il
codice mostrato sulla pagina resta corto e leggibile.
"""

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import AgoGPT, Blk, RMSNorm, CausalSelfAttention, SwiGLU
from llm_anatomy.tokenizer import CharTokenizer
from llm_anatomy import data, sample, train, utils

__all__ = [
    "AgoGPTConfig",
    "AgoGPT",
    "Blk",
    "RMSNorm",
    "CausalSelfAttention",
    "SwiGLU",
    "CharTokenizer",
    "data",
    "sample",
    "train",
    "utils",
]
