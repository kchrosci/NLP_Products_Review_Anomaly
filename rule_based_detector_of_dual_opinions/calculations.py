import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tabulate import tabulate
from collections import Counter
from config_loader import ConfigLoader

config = ConfigLoader("config.json")
config.load_config()
nlp = spacy.load(config.get_value("spacy"))


def load_opinions():
    # Ten tworzy kombinacje
    opinions = pd.read_csv(config.get_value("anomaly_opinions"), sep=',')
    sentences = opinions['content'].tolist()

    # Ten zmieniać jeśli sie chce sprawdzać różne sety
    opinions = pd.read_csv(config.get_value("anomaly_opinions"), sep=',')
    opinions_list = [(row['content'], row['doubleQuality'])
                     for _, row in opinions.iterrows()]

    random.shuffle(opinions_list)
    return sentences, opinions_list


def unique_words_in_csv(sentences):
    lem_counter = Counter()
    for content in sentences:
        doc = nlp(content)

        for token in doc:
            if token.is_alpha and (not token.is_stop or token.lower_ == "niż" or token.lower_ == "od"):
                lem_counter[token.lemma_.lower()] += 1

    lemmas_count = list(zip(lem_counter.keys(), lem_counter.values()))
    sorted_lemmas_count = sorted(
        lemmas_count, key=lambda x: x[1], reverse=True)
    return sorted_lemmas_count


def create_combinations(lemmas):
    combinations = set()
    for lemma1 in config.get_value("manually_selected_lemmas"):
        for lemma2 in lemmas:
            combination = tuple(sorted([lemma1, lemma2]))
            combinations.add(combination)

    for lemma1 in lemmas:
        for lemma2 in lemmas:
            if lemma1 != lemma2:
                combination = tuple(sorted([lemma1, lemma2]))
                combinations.add(combination)

    combinations = {(lemma1, lemma2)
                    for lemma1, lemma2 in combinations if lemma1 != lemma2}

    return combinations


def prepare_lemmas(lemmas_count):
    lemmas = []
    for lemma, count in lemmas_count:
        if count > 5:
            lemmas.append(lemma)
    return lemmas


def rule_based_double_quality_search(opinions, combinations):

    total_sentences = len(opinions)
    true_positives = config.get_value("true_positives")
    false_positives = config.get_value("false_positives")
    true_negatives = config.get_value("true_negatives")
    false_negatives = config.get_value("false_negatives")

    for txt in opinions:
        # Przetworzenie zdania na tokeny
        doc = nlp(txt[0])

        # Tworzenie listy lematów zdania
        lemmas = [token.lemma_.lower() for token in doc]

        # Tworzenie kombinacji lematów zdania
        sentence_combinations = set()
        for lemma1, lemma2 in combinations:
            if lemma1 in lemmas and lemma2 in lemmas:
                sentence_combinations.add((lemma1, lemma2))

        # Sprawdzenie liczby kombinacji
        if len(sentence_combinations) > 10:
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
