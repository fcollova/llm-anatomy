"""Dal file di testo alle coppie domanda-risposta.

Il modello impara una cosa sola: dato un pezzo di testo, indovinare il
carattere che viene dopo. Questo script mostra come dalle ricette
dell'Artusi si ricavino centinaia di migliaia di esercizi di quel tipo, e
perche' non serva nessuno a scrivere le risposte.

Si esegue con `python3 scripts/05_01_dati_e_batching.py`
e dura circa 5 secondi.

Da provare: cambia il punto di taglio fra studio e controllo e rifai il conto.
"""

import torch

from llm_anatomy.data import (NOME_CORPUS, carica_testo, get_batch,
                              prepara_split)
from llm_anatomy.utils import (byte_leggibili, dec, numero, prepara, stampa_shape,
                               tabella, titolo, verifica)

prepara()

titolo("Le ricette dell'Artusi")
testo = carica_testo()
print(f"  file:       data/{NOME_CORPUS}")
print(f"  caratteri:  {numero(len(testo))}")
print(f"  distinti:   {len(set(testo))}")
print(f"  su disco:   {byte_leggibili(len(testo.encode('utf-8')))}")
print()
print("  prime cinque righe:")
for riga in testo.split("\n")[:5]:
    print(f"    | {riga}")

titolo("Novantatre' simboli, e nient'altro")
tok, dati_train, dati_val = prepara_split()
print(f"  dimensione: {tok.vocab_size}")
visibili = "".join(c if c not in "\n\r\t" else " " for c in tok.itos)
print(f"  simboli:    {visibili!r}")
print()
frase = "Prima parola."
codificata = tok.encode(frase)
print(f"  encode({frase!r}) = {codificata}")
print(f"  decode(...)      = {tok.decode(codificata)!r}")
verifica(tok.decode(tok.encode(testo[:5000])) == testo[:5000],
         "encode e decode sono l'una l'inversa dell'altra sul corpus")

# Il taglio e' per posizione e non a caso: se le due meta' fossero mescolate,
# una finestra di contesto potrebbe contenere testo di studio e testo di
# controllo insieme, e il controllo non controllerebbe piu' niente.
titolo("Testo per studiare, testo per controllare")
n_tot = len(dati_train) + len(dati_val)
tabella(
    ["insieme", "token", "quota"],
    [["addestramento", numero(len(dati_train)), f"{100 * len(dati_train) / n_tot:.0f}%"],
     ["validazione", numero(len(dati_val)), f"{100 * len(dati_val) / n_tot:.0f}%"]],
)
print("  tagliare per posizione e non a caso impedisce che una finestra di")
print("  contesto contenga token di entrambi gli insiemi")

titolo("La risposta e' la domanda, spostata di uno")
# Finestre minuscole -- quattro sequenze da otto caratteri -- per poterle
# stampare tutte. Nell'addestramento vero sono sedici da centoventotto.
B, T = 4, 8
gen = torch.Generator().manual_seed(1337)
x, y = get_batch(dati_train, B, T, gen)
stampa_shape("x", x)
stampa_shape("y", y)
print()
print("  prima sequenza del batch:")
print(f"    x = {x[0].tolist()}")
print(f"    y = {y[0].tolist()}")
print(f"    x testo: {tok.decode(x[0])!r}")
print(f"    y testo: {tok.decode(y[0])!r}")
verifica(torch.equal(x[0, 1:], y[0, :-1]),
         "y e' x sfalsato di una posizione")

titolo("Ogni posizione e' un esercizio")
# Il punto meno ovvio del capitolo: una finestra da otto caratteri non e' un
# esercizio, sono otto esercizi, perche' la maschera causale fa in modo che
# ogni posizione veda solo cio' che sta prima di lei.
print("  ogni posizione della finestra e' un esempio a se':")
for t in range(4):
    contesto = tok.decode(x[0, :t + 1])
    bersaglio = tok.decode([y[0, t]])
    print(f"    contesto {contesto!r:<14} -> bersaglio {bersaglio!r}")
print(f"  un batch {B}x{T} contiene quindi {B * T} coppie contesto-bersaglio")

titolo("Quante volte rilegge il libro")
righe = []
for passi in (500, 1000, 2500):
    token = passi * 16 * 128          # sedici finestre da 128 a ogni passo
    righe.append([numero(passi), numero(token), dec(token / len(dati_train))])
tabella(["passi", "token visti", "epoche sul corpus"], righe)
volte = 2500 * 16 * 128 / len(dati_train)
print(f"  con B=16 e T=128, i 2500 passi dell'addestramento rileggono "
      f"il testo {dec(volte, 1)}")
print("  volte: abbastanza per imparare l'ortografia e le formule che tornano")
print("  in ogni ricetta, troppo poco per sapere le ricette a memoria")
