# Záznam do deníku – Tvorba FFNN pomocí Keras (sekvenční model)

## Zadání
- Vyzkoušet nástroje knihovny Keras pro tvorbu neuronových sítí.
- Aplikovat na úlohu klasifikace (Iris dataset).
- Připravit data pro trénování, validaci a testování.
- Navrhnout 5 různých topologií, každou trénovat 10krát.
- Vytvořit boxplot graf výsledků a uložit nejlepší model.

---

## Použitá knihovna
- **TensorFlow 2.19** s podporou GPU (CUDA 12.2, cuDNN 8.9)
- **Keras Sequential API**

---

## Úloha
Byla použita klasifikační úloha s datovou sadou **Iris**.  
Cílem bylo předpovědět druh květiny na základě 4 měřených znaků (délka a šířka kališního a korunního lístku).

---

## Příprava dat
- Normalizace vstupních dat pomocí `StandardScaler`.
- One-hot encoding cílových tříd pomocí `LabelBinarizer`.
- Rozdělení na trénovací (60 %), validační (20 %) a testovací (20 %) sady.

---

## Navržené topologie neuronových sítí
| Topologie | Popis |
|:----------|:------|
| Topologie 1 | 8 neuronů → 3 výstupy |
| Topologie 2 | 16 neuronů → 8 neuronů → 3 výstupy |
| Topologie 3 | 32 neuronů → 16 neuronů → 8 neuronů → 3 výstupy |
| Topologie 4 | 64 neuronů → 32 neuronů → 16 neuronů → 8 neuronů → 3 výstupy |
| Topologie 5 | 128 neuronů → 64 neuronů → 32 neuronů → 16 neuronů → 8 neuronů → 3 výstupy |

Každý model byl trénován 10krát s náhodnou inicializací vah.

---

## Výsledky
- Pro každou topologii byla zaznamenána validační přesnost po trénování.
- Výsledky byly shrnuty pomocí boxplot grafu:

![Boxplot validační přesnosti](boxplot_val_accuracy.png)

---

## Nejlepší model
- Nejvyšší medián validační přesnosti dosáhla topologie **[dosaď nejlepší]**.
- Tento model byl následně natrénován na celé trénovací a validační sadě a uložen do souboru:



### Vyhodnocení na testovacích datech:
- **Test Loss:** `[doplnit hodnotu]`
- **Test Accuracy:** `[doplnit hodnotu]`

---

## Závěr
- Bylo ověřeno, že architektura neuronové sítě výrazně ovlivňuje výslednou přesnost modelu.
- Nejlepší topologie dosáhla testovací přesnosti **[doplnit]%**.
- Uložení modelu a grafu umožňuje další analýzy a použití v budoucnu.

---
