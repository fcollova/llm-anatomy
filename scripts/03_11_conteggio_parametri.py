"""Contare i parametri a mente.

C'e' una formula che stima i parametri di un Transformer senza aprire il
codice, e sbaglia di pochissimo. Vale la pena impararla, e vale ancora di piu'
sapere quando smette di valere.

Si esegue con `python3 scripts/03_11_conteggio_parametri.py`
e dura circa 15 secondi.

Da provare: aggiungi una configurazione con vocabolario da 128.000 simboli.
"""

from llm_anatomy.config import AgoGPTConfig
from llm_anatomy.model import AgoGPT
from llm_anatomy.utils import dec, numero, prepara, tabella, titolo, verifica

prepara()

titolo("Da dove viene il dodici")
print("  attenzione: Wq, Wk, Wv, Wo, quattro matrici d x d      ->  4 d^2")
print("  FFN classica: salita d x 4d e discesa 4d x d           ->  8 d^2")
print("  totale per blocco                                      -> 12 d^2")
print("  moltiplicato per N blocchi                             -> 12 N d^2")
print("  restano fuori: embedding, norme, e la testa se non condivisa")

titolo("Cinque modelli, cinque conti")
righe = []
for n_layer, d_model, n_head in ((2, 128, 4), (4, 192, 6), (6, 384, 6),
                                 (12, 768, 12), (24, 1024, 16)):
    cfg = AgoGPTConfig(vocab_size=93, n_layer=n_layer, d_model=d_model,
                        n_head=n_head, act="gelu", bias=False)
    m = AgoGPT(cfg)
    misurati = m.numero_parametri(senza_embedding=True)
    formula = 12 * n_layer * d_model ** 2
    scarto = 100 * (misurati - formula) / formula
    righe.append([
        f"L{n_layer} d{d_model}",
        numero(misurati),
        numero(formula),
        ("+" if scarto >= 0 else "") + dec(scarto) + "%",
    ])
tabella(["modello", "misurati (senza emb.)", "12 N d^2", "scarto"], righe)
print("  lo scarto e' quello delle sole norme: due vettori di d elementi per")
print("  blocco piu' uno finale, cioe' un termine lineare in d che scompare")
print("  appena il modello cresce")

titolo("Verifica")
cfg = AgoGPTConfig(vocab_size=93, n_layer=12, d_model=768, n_head=12,
                    act="gelu", bias=False)
m = AgoGPT(cfg)
misurati = m.numero_parametri(senza_embedding=True)
formula = 12 * 12 * 768 ** 2
verifica(abs(misurati - formula) / formula < 0.01,
         "la formula 12 N d^2 e' esatta entro l'1% sulla configurazione da 12 blocchi")

titolo("Dove smette di valere")
righe = []
for nome, kwargs in (
    ("classica (riferimento)", dict(act="gelu")),
    ("SwiGLU", dict(act="swiglu")),
    ("GQA a 4 gruppi", dict(act="gelu", n_kv_head=4)),
    ("SwiGLU + GQA", dict(act="swiglu", n_kv_head=4)),
):
    cfg = AgoGPTConfig(vocab_size=93, n_layer=12, d_model=768, n_head=12,
                        bias=False, **kwargs)
    m = AgoGPT(cfg)
    misurati = m.numero_parametri(senza_embedding=True)
    rapporto = misurati / formula
    righe.append([nome, cfg.d_ff, numero(misurati), dec(rapporto, 3)])
tabella(["variante", "d_ff", "parametri", "rispetto a 12 N d^2"], righe)
print("  SwiGLU resta vicinissimo, perche' 8/3 e' scelto apposta per questo;")
print("  GQA toglie parametri dall'attenzione e porta il rapporto sotto uno")

titolo("Quello che la formula ignora")
righe = []
for vocab in (93, 8000, 32000, 128000):
    for n_layer, d_model in ((4, 192), (12, 768)):
        cfg = AgoGPTConfig(vocab_size=vocab, n_layer=n_layer, d_model=d_model,
                            n_head=max(2, d_model // 64), act="gelu", bias=False)
        m = AgoGPT(cfg)
        emb = cfg.vocab_size * cfg.d_model
        tot = m.numero_parametri()
        righe.append([numero(vocab), f"L{n_layer} d{d_model}", numero(tot),
                      f"{100 * emb / tot:.1f}%".replace(".", ",")])
tabella(["vocabolario", "modello", "parametri totali", "quota dell'embedding"],
        righe)
print("  con un vocabolario reale e un modello piccolo l'embedding e' meta'")
print("  del totale: e' il motivo per cui il conteggio 'senza embedding' e'")
print("  quello che compare nelle leggi di scala")
