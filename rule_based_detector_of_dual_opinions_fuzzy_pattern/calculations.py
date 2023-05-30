import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tabulate import tabulate
from spacy.matcher import Matcher
from config_loader import ConfigLoader

config = ConfigLoader("config.json")
config.load_config()
nlp = spacy.load(config.get_value("spacy"))
# Obiekt PhraseMatcher
matcher = Matcher(nlp.vocab)


def load_opinions():
    opinions = pd.read_csv(config.get_value("anomaly_opinions"), sep=',')
    # Tworzenie listy krotek
    opinions_list = [(row['content'], row['doubleQuality'])
                     for _, row in opinions.iterrows()]
    random.shuffle(opinions_list)
    return opinions_list


def get_patterns():
    patterns = config.get_value("patterns")
    matcher.add("DoubleQuality", patterns)
    return patterns


def rule_based_double_quality_search(dataset):

    total_sentences = len(dataset)
    true_positives = config.get_value("true_positives")
    false_positives = config.get_value("false_positives")
    true_negatives = config.get_value("true_negatives")
    false_negatives = config.get_value("false_negatives")

    for txt in dataset:
        # Tekst wejściowy jako obiekt typu Document
        doc = nlp(txt[0])

        # Wywołanie metody matcher na obiekcie doc. Zwraca obiekty typu Span
        matches = matcher(doc)

        if len(matches) > 0:
            if txt[1] == 1:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if txt[1] == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    # Tworzenie listy z danymi
    data = [
        ["Liczba zdań ze zbioru:", total_sentences],
        ["TP:", true_positives],
        ["FN:", false_negatives],
        ["FP:", false_positives],
        ["TN:", true_negatives]
    ]

    # Obliczenie metryk
    if (true_positives + false_positives) != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if (true_positives + false_negatives) != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    if (true_positives + true_negatives + false_positives + false_negatives) != 0:
        accuracy = (true_positives + true_negatives) / (true_positives +
                                                        true_negatives + false_positives + false_negatives)
    else:
        accuracy = 0.0

    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    if (true_positives + true_negatives + false_positives + false_negatives) != 0:
        loss = (false_positives + false_negatives) / (true_positives +
                                                      true_negatives + false_positives + false_negatives)
    else:
        loss = 0.0

    # Dodanie metryk do listy danych
    data.extend([
        ["Miary:"],
        ["Precision (precyzja):", precision],
        ["Recall (czułość):", recall],
        ["Accuracy (dokładność):", accuracy],
        ["Loss (strata):", loss],
        ["F1 Score (miara F1):", f1_score]
    ])

    # Wypisanie danych w postaci tabeli
    print(tabulate(data, headers=["Miary", "Wartości"], tablefmt="fancy_grid"))

    confusion_matrix = [[true_positives, false_negatives],
                        [false_positives, true_negatives]]

   # Tworzenie etykiet dla macierzy pomyłek
    labels = [["Pozytywna", "Negatywna"], ["Pozytywna", "Negatywna"]]

    # Wykreślenie Confusion Matrix z etykietami
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues",
                fmt="d", xticklabels=labels[0], yticklabels=labels[1])
    plt.xlabel("Klasa predykowana")
    plt.ylabel("Klasa rzeczywista")
    plt.title("Tablica pomyłek")
    plt.show()
