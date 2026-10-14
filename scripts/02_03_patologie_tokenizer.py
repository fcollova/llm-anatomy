"""Le stranezze che vengono da li', e che si vedono ogni giorno.

Se il modello riceve simboli gia' fusi e non lettere, alcune cose diventano
difficili per motivi che non hanno niente a che fare con l'intelligenza:
contare le lettere di una parola, distinguere una parola con lo spazio davanti
da una senza, fare i conti con numeri lunghi. Qui si misurano.

Si esegue con `python3 scripts/02_03_patologie_tokenizer.py`
e dura circa 5 secondi.

Da provare: aggiungi una parola italiana lunga e conta i suoi pezzi.
"""

from llm_anatomy.data import carica_testo
from llm_anatomy.tokenizer import addestra_bpe
from llm_anatomy.utils import dec, numero, prepara, tabella, titolo, verifica

prepara()

tok = addestra_bpe(carica_testo()[:400_000], vocab_size=1200)
print(f"  tokenizer BPE addestrato: {tok.get_vocab_size()} simboli")


def pezzi(s: str) -> list[str]:
    """Segmentazione, con i simboli non ASCII resi come sequenze di escape.

    Il marcatore di spazio del ByteLevel e' U+0120: stamparlo cosi' com'e'
    renderebbe l'output non componibile da LaTeX e illeggibile su terminali
    senza il font adatto.
    """
    return [p.encode("ascii", "backslashreplace").decode()
            for p in tok.encode(s).tokens]


titolo("Perche' sbaglia a contare le lettere")
for parola, lettera in (("prezzemolo", "z"), ("sottilissime", "s"),
                        ("mescolatelo", "e")):
    p = pezzi(parola)
    print(f"  {parola!r:<16} -> {p}")
    quante = parola.count(lettera)
    print(f"  {'':<16}    {quante} lettere {lettera!r} sparse in {len(p)} "
          f"token, mescolate ad altre")
print("  chiedere quante lettere ci sono significa chiedere di contare dentro")
print("  simboli che il modello riceve gia' fusi")

titolo("Lo spazio davanti cambia tutto")
righe = []
for parola in ("Burro", "burro"):
    for prefisso, nome in ((" ", "con spazio"), ("", "senza spazio")):
        p = pezzi(prefisso + parola)
        righe.append([f"{prefisso + parola!r}", nome, len(p), " ".join(p)])
tabella(["stringa", "forma", "n", "token"], righe)
print("  la stessa parola ha identita' diverse a seconda del contesto:")
print("  e' la ragione per cui i prompt che finiscono con uno spazio si")
print("  comportano in modo diverso da quelli che non ce l'hanno")

titolo("I numeri, che nessuno taglia bene")
righe = []
for n in ("7", "42", "365", "1265", "1000000", "3.14159"):
    p = pezzi(n)
    righe.append([n, len(p), " ".join(repr(x) for x in p)])
tabella(["numero", "token", "segmentazione"], righe)
print("  cifre raggruppate in modo arbitrario: la posizione delle unita' non")
print("  e' una proprieta' visibile al modello, e l'aritmetica in colonna")
print("  richiede allora di ricostruirla dai pezzi")

titolo("Costa di piu' scrivere in altre lingue")
campioni = {
    "italiano": "Mettete al fuoco il burro e fate bollire il latte adagio.",
    "inglese": "Put the butter on the fire and let the milk boil slowly.",
    "tedesco": "Die Butter aufs Feuer stellen und die Milch langsam kochen.",
}
righe = []
for lingua, frase in campioni.items():
    n_tok = len(pezzi(frase))
    n_parole = len(frase.split())
    righe.append([lingua, len(frase), n_tok, n_parole, dec(n_tok / n_parole)])
tabella(["lingua", "caratteri", "token", "parole", "token per parola"], righe)
inglese = len(pezzi(campioni["inglese"])) / len(campioni["inglese"].split())
italiano = len(pezzi(campioni["italiano"])) / len(campioni["italiano"].split())
print(f"  l'inglese costa {dec(inglese / italiano)} volte l'italiano a "
      "parita' di")
print("  contenuto: stessa finestra di contesto, meno testo dentro, e un")
print("  conto a token che cresce nella stessa proporzione")
verifica(inglese > italiano,
         "il costo per parola e' maggiore nella lingua che il corpus non "
         "contiene")

titolo("Una causa sola")
print("  il modello riceve identificatori interi, non testo. Ogni patologia")
print("  qui sopra e' la stessa osservazione vista da un lato diverso: cio'")
print("  che non e' un token non e' un'unita' su cui il modello possa operare")
