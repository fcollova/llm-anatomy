# llm-anatomy

Il codice di *Anatomia di un LLM*: un modello linguistico piccolo, costruito da
zero in PyTorch e addestrato su un processore, senza scheda grafica.

Il modello si chiama **AgoGPT** e sta in `llm_anatomy/`. Si addestra in circa
cinque minuti sulle ricette de *La scienza in cucina e l'arte di mangiar bene*
di Pellegrino Artusi, e poi scrive ricette che sembrano ricette e non si
possono cucinare.

## Il modo più rapido: i notebook

Ogni script esiste anche come notebook, già eseguito: si legge qui su GitHub
senza installare niente, e si esegue su Colab con un clic. L'indice, capitolo
per capitolo, è in [`notebooks/`](notebooks/).

## Sul proprio computer

Serve Python 3.10 o successivo.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
python3 scripts/03_04_maschere.py
```

`pip install -e .` va dato una volta: dice a Python dove sta la cartella del
modello. Lo script di addestramento è `scripts/05_02_pretrain_agogpt.py`; gli
script che usano il modello addestrato lo cercano in `checkpoints/agogpt.pt`,
che quello produce. In alternativa, i soli pesi sono allegati alla release
`modello-v0.1`.

## Che cosa c'è

```
scripts/       gli esperimenti, uno per file, nell'ordine del libro
llm_anatomy/   il modello: attenzione, blocco, addestramento, generazione
notebooks/     gli stessi esperimenti, a celle
data/          il ricettario su cui il modello si addestra
```

Ogni numero che gli script stampano è quello stampato nel libro: tutto parte
dallo stesso seme, e l'addestramento dà lo stesso modello byte per byte. Fanno
eccezione i tempi, che dipendono dalla macchina.

## Il testo

Il ricettario è di pubblico dominio: l'Artusi morì nel 1911. Il file viene
dalla 25ª edizione, del 1922, ridotto alle sole ricette.

## Licenza

Il codice è distribuito con licenza MIT: vedi `LICENSE`.
