"""Configurazione del mini-modello: un unico oggetto passato a tutto il resto."""

from dataclasses import dataclass, asdict, field


def multiplo_di(x: int, base: int) -> int:
    """Arrotonda x al multiplo di base piu' vicino verso l'alto."""
    return ((x + base - 1) // base) * base


@dataclass
class AgoGPTConfig:
    """Iperparametri del modello.

    I default riproducono il modello del libro: 4 blocchi,
    d_model 192, 6 teste, contesto 128. Sono ~1,8 M parametri, la taglia piu'
    grande che su un portatile si addestra in pochi minuti.
    """

    vocab_size: int = 93
    block_size: int = 128
    n_layer: int = 4
    d_model: int = 192
    n_head: int = 6
    n_kv_head: int | None = None      # None: MHA; 1: MQA; in mezzo: GQA
    d_ff: int | None = None           # None => 8/3 * d_model, multiplo di 32
    dropout: float = 0.0
    norm: str = "rms"                 # "rms" | "layer"
    act: str = "swiglu"               # "swiglu" | "gelu" | "relu"
    pos: str = "rope"                 # o "learned", "sinusoidal", "none"
    rope_theta: float = 10000.0
    rope_scaling: float = 1.0         # >1: posizioni interpolate
    tie_weights: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model {self.d_model} non divisibile "
                             f"per n_head {self.n_head}")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head {self.n_head} non multiplo "
                             f"di n_kv_head {self.n_kv_head}")
        if self.d_ff is None:
            # SwiGLU ha tre matrici invece di due: per mantenere lo stesso
            # numero di parametri della FFN classica d_ff scende da 4d a 8/3 d.
            fattore = 8 / 3 if self.act == "swiglu" else 4
            self.d_ff = multiplo_di(int(fattore * self.d_model), 32)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def as_dict(self) -> dict:
        return asdict(self)

    def descrizione(self) -> str:
        return (
            f"L={self.n_layer} d={self.d_model} "
            f"h={self.n_head}/{self.n_kv_head}kv "
            f"d_head={self.d_head} d_ff={self.d_ff} T={self.block_size} "
            f"norm={self.norm} act={self.act} pos={self.pos}"
        )
