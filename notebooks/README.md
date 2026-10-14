# I notebook di *Anatomia di un LLM*

Ogni script del libro, tagliato in celle: una per ogni blocco che lo script separa con un titolo. Servono a cambiare un numero e rieseguire un pezzo solo. Sono generati dagli script della cartella `scripts/`: una correzione va fatta lì.

Sono salvati già eseguiti, con le uscite del libro: si possono leggere senza installare niente.

Per eseguirli senza installare niente c'è Colab: il collegamento è nell'ultima colonna, e la prima cella di ogni notebook scarica da sé il codice e il ricettario.

Cinque notebook usano il modello addestrato: se manca, ne scaricano i pesi. Sono segnati con una nota.

## Capitolo 3. I mattoni

| Notebook | Che cosa mostra | Durata | Colab |
|---|---|---|---|
| [`01_01_tensori_e_shape`](01_01_tensori_e_shape.ipynb) | tensori | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_01_tensori_e_shape.ipynb) |
| [`01_02_prodotto_scalare`](01_02_prodotto_scalare.ipynb) | prodotto scalare | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_02_prodotto_scalare.ipynb) |
| [`01_09_moltiplicare_matrici`](01_09_moltiplicare_matrici.ipynb) | moltiplicare matrici | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_09_moltiplicare_matrici.ipynb) |
| [`01_03_neurone_e_strato`](01_03_neurone_e_strato.ipynb) | uno strato | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_03_neurone_e_strato.ipynb) |
| [`01_04_non_linearita`](01_04_non_linearita.ipynb) | non linearita' | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_04_non_linearita.ipynb) |
| [`01_05_autograd`](01_05_autograd.ipynb) | gradienti | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_05_autograd.ipynb) |
| [`01_06_training_loop`](01_06_training_loop.ipynb) | il ciclo | 10 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_06_training_loop.ipynb) |
| [`01_07_softmax_entropia`](01_07_softmax_entropia.ipynb) | softmax e perdita | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_07_softmax_entropia.ipynb) |

## Capitolo 4. Dal testo ai numeri

| Notebook | Che cosa mostra | Durata | Colab |
|---|---|---|---|
| [`02_01_bpe_da_zero`](02_01_bpe_da_zero.ipynb) | il tokenizer, a mano | 15 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/02_01_bpe_da_zero.ipynb) |
| [`02_03_patologie_tokenizer`](02_03_patologie_tokenizer.ipynb) | le stranezze | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/02_03_patologie_tokenizer.ipynb) |
| [`02_04_embedding_geometria`](02_04_embedding_geometria.ipynb) | embedding ¹ | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/02_04_embedding_geometria.ipynb) |
| [`02_06_rope`](02_06_rope.ipynb) | la posizione | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/02_06_rope.ipynb) |

## Capitolo 5. L'attenzione e il blocco

| Notebook | Che cosa mostra | Durata | Colab |
|---|---|---|---|
| [`03_01_attenzione_da_zero`](03_01_attenzione_da_zero.ipynb) | attenzione, a mano | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_01_attenzione_da_zero.ipynb) |
| [`03_02_verifica_sdpa`](03_02_verifica_sdpa.ipynb) | attenzione, dalla libreria | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_02_verifica_sdpa.ipynb) |
| [`03_03_perche_radice_dk`](03_03_perche_radice_dk.ipynb) | la radice di d_k | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_03_perche_radice_dk.ipynb) |
| [`03_04_maschere`](03_04_maschere.ipynb) | le maschere | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_04_maschere.ipynb) |
| [`03_05_multihead`](03_05_multihead.ipynb) | piu' teste | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_05_multihead.ipynb) |
| [`03_07_ffn_attivazioni`](03_07_ffn_attivazioni.ipynb) | la rete del blocco | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_07_ffn_attivazioni.ipynb) |
| [`03_08_normalizzazione`](03_08_normalizzazione.ipynb) | normalizzare | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_08_normalizzazione.ipynb) |
| [`03_09_blocco_completo`](03_09_blocco_completo.ipynb) | il blocco | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_09_blocco_completo.ipynb) |
| [`03_10_modello_completo`](03_10_modello_completo.ipynb) | il modello intero | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_10_modello_completo.ipynb) |
| [`03_11_conteggio_parametri`](03_11_conteggio_parametri.ipynb) | contare i parametri | 15 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/03_11_conteggio_parametri.ipynb) |

## Capitolo 6. Addestrarlo

| Notebook | Che cosa mostra | Durata | Colab |
|---|---|---|---|
| [`05_01_dati_e_batching`](05_01_dati_e_batching.ipynb) | dati e batch | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/05_01_dati_e_batching.ipynb) |
| [`05_02_pretrain_agogpt`](05_02_pretrain_agogpt.ipynb) | addestramento | 5 min | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/05_02_pretrain_agogpt.ipynb) |
| [`05_04_schedule_lr`](05_04_schedule_lr.ipynb) | il passo, nel tempo | 35 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/05_04_schedule_lr.ipynb) |

## Capitolo 7. Generare

| Notebook | Che cosa mostra | Durata | Colab |
|---|---|---|---|
| [`01_08_memory_bound`](01_08_memory_bound.ipynb) | perche' e' lento | 10 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/01_08_memory_bound.ipynb) |
| [`06_01_generazione_ingenua`](06_01_generazione_ingenua.ipynb) | generare, nel modo ovvio ¹ | 10 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/06_01_generazione_ingenua.ipynb) |
| [`06_02_kv_cache`](06_02_kv_cache.ipynb) | la cache ¹ | 10 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/06_02_kv_cache.ipynb) |
| [`06_03_campionatori`](06_03_campionatori.ipynb) | scegliere il carattere ¹ | 5 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/06_03_campionatori.ipynb) |
| [`06_04_prefill_decode`](06_04_prefill_decode.ipynb) | le due fasi ¹ | 10 s | [apri](https://colab.research.google.com/github/fcollova/llm-anatomy/blob/main/notebooks/06_04_prefill_decode.ipynb) |

¹ usa il modello addestrato da `05_02_pretrain_agogpt`.

Le durate sono quelle della macchina su cui il libro è stato costruito, con otto processi di calcolo. Con due processi, che è quanto offre un servizio gratuito nel browser, l'addestramento dura 8,4 minuti invece di cinque, e produce lo stesso modello byte per byte: i numeri del libro non dipendono da quanti processi ha la macchina.
