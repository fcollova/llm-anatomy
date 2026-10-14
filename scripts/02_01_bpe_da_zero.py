"""Il modello non vede lettere. Vede simboli, e se li e' scelti da solo.

Prima di arrivare al modello, il testo va ridotto a numeri. Il modo usato
ovunque non spezza per parole ne' per lettere: parte dalle lettere e poi fonde
ripetutamente la coppia piu' frequente, finche' il vocabolario e' pieno. Si
chiama BPE, e in questa pagina lo costruiamo da zero.

Si esegue con `python3 scripts/02_01_bpe_da_zero.py`
e dura circa 15 secondi.

Da provare: porta i merge a mille e guarda la curva dei caratteri per token appiattirsi.
"""

from collections import Counter

from llm_anatomy.data import carica_testo
from llm_anatomy.utils import numero, prepara, tabella, titolo, verifica

prepara()

# Un corpus ridotto: l'algoritmo e' quadratico nel numero di merge e qui
# interessa vederlo lavorare, non produrre un tokenizer da usare davvero.
testo = carica_testo()[:200_000]
N_MERGE = 335

titolo("Si parte dalle lettere")
simboli = sorted(set(testo))
vocabolario = {c: i for i, c in enumerate(simboli)}
sequenza = [vocabolario[c] for c in testo]
print(f"  corpus:      {numero(len(testo))} caratteri")
print(f"  vocabolario: {len(vocabolario)} simboli")
print(f"  sequenza:    {numero(len(sequenza))} token")


def conta_coppie(seq: list[int]) -> Counter:
    """Conta ogni coppia di token adiacenti. E' l'unica statistica che serve."""
    return Counter(zip(seq, seq[1:]))


def fondi(seq: list[int], coppia: tuple[int, int], nuovo: int) -> list[int]:
    """Sostituisce ogni occorrenza della coppia con un solo token nuovo."""
    fuori, i = [], 0
    a, b = coppia
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            fuori.append(nuovo)
            i += 2
        else:
            fuori.append(seq[i])
            i += 1
    return fuori


titolo("Le prime dieci fusioni")
inverso = {i: c for c, i in vocabolario.items()}
merge: list[tuple[tuple[int, int], int]] = []
righe = []
for passo in range(N_MERGE):
    conteggi = conta_coppie(sequenza)
    if not conteggi:
        break
    coppia, frequenza = conteggi.most_common(1)[0]
    nuovo = len(inverso)
    testo_nuovo = inverso[coppia[0]] + inverso[coppia[1]]
    inverso[nuovo] = testo_nuovo
    merge.append((coppia, nuovo))
    prima = len(sequenza)
    sequenza = fondi(sequenza, coppia, nuovo)
    if passo < 10:
        righe.append([
            passo + 1,
            f"{inverso[coppia[0]]!r} + {inverso[coppia[1]]!r}",
            repr(testo_nuovo),
            numero(frequenza),
            numero(prima - len(sequenza)),
        ])
tabella(["merge", "coppia", "nuovo simbolo", "occorrenze", "token risparmiati"],
        righe)
print("  i primi merge sono sequenze frequentissime e senza significato:")
print("  vocale piu' spazio, virgola piu' spazio, sillabe")

# Quanti dei merge appresi scavalcano un confine di parola: e' esattamente cio'
# che la pre-tokenizzazione di 02-02 rende impossibile. Lo splitter a regex
# ammette lo spazio solo in testa a un pezzo, e le sequenze di soli spazi come
# pezzo a se': tutto il resto e' un merge che unisce due parole diverse.
def a_cavallo(t: str) -> bool:
    resto = t.lstrip(" \n")           # l'eventuale spazio iniziale e' lecito
    return any(c in " \n" for c in resto)


scavalcano = [t for i, t in inverso.items()
              if i >= len(vocabolario) and a_cavallo(t)]
print(f"  merge che scavalcano un confine di parola: {len(scavalcano)}"
      f" su {N_MERGE}")
print(f"  i primi: {[t for t in scavalcano[:6]]}")
print("  in un tokenizer reale non esisterebbero: lo splitter spezza il testo")
print("  prima dei merge e nessuna fusione attraversa i suoi confini")

titolo("Piu' simboli, sequenze piu' corte")
righe = []
for soglia in (0, 50, 150, N_MERGE):
    # Ricostruzione da capo fino al merge indicato: costa, ma rende visibile
    # la curva senza doversi fidare di un conteggio incrementale.
    s = [vocabolario[c] for c in testo]
    for coppia, nuovo in merge[:soglia]:
        s = fondi(s, coppia, nuovo)
    righe.append([
        soglia,
        len(vocabolario) + soglia,
        numero(len(s)),
        f"{len(testo) / len(s):.2f}".replace(".", ","),
    ])
tabella(["merge", "vocabolario", "token", "caratteri per token"], righe)

titolo("Che cosa e' diventato un simbolo solo")
lunghi = sorted(((len(t), i, t) for i, t in inverso.items()
                 if i >= len(vocabolario)),
                reverse=True)[:12]
print("  i dodici simboli piu' lunghi del vocabolario finale:")
for lung, idx, t in lunghi:
    print(f"    id {idx:4d}  {t!r}")

titolo("Avanti e indietro, senza perdere niente")
def decodifica(seq: list[int]) -> str:
    return "".join(inverso[i] for i in seq)

verifica(decodifica(sequenza) == testo,
         "la sequenza fusa decodifica esattamente al corpus di partenza")
print(f"  compressione finale: {len(testo)} caratteri -> {len(sequenza)} token")
print(f"  rapporto: {len(testo) / len(sequenza):.2f} caratteri per token"
      .replace(".", ","))
print("  un BPE reale su miliardi di caratteri arriva a 4 circa: qui il corpus")
print("  e' piccolo e i merge disponibili si esauriscono prima")
