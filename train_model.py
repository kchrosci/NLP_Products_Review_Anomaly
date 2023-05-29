import re
import spacy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from config_loader import ConfigLoader
from autoencoder import Autoencoder

# Wczytywanie configu
config = ConfigLoader("config.json")
config.load_config()

# Inicjalizacja modelu spaCy
nlp = spacy.load(config.get_value("spacy"))

# Ustawienie seed'a
SEED = config.get_value("seed")
torch.manual_seed(SEED)


# Klasa Dataset do przechowywania danych opinii
class OpinionDataset(Dataset):
    def __init__(self, opinions):
        self.opinions = opinions

    def __len__(self):
        return len(self.opinions)

    def __getitem__(self, idx):
        doc = nlp(preprocess_review(self.opinions[idx]))
        sequence = np.array([token.vector for token in doc])
        sequence = torch.tensor(sequence, dtype=torch.float32)
        #sequence = nn.functional.normalize(sequence, p=float('inf'), dim=1)
        return sequence


def load_opinions():

    # Wczytanie opinii
    anomaly_opinions = read_csv(config.get_value("anomaly_opinions"), sep=',')
    normal_opinions = read_csv(config.get_value("normal_opinions"), sep=',')
    anomaly_opinions = anomaly_opinions[config.get_value("content")].values.tolist()
    normal_opinions = normal_opinions[config.get_value("content")].values.tolist()

    # Do szybkich testów
    normal_opinions = normal_opinions[:500]
    return normal_opinions, anomaly_opinions


def split_data(normal_opinions, anomaly_opinions):
    # Podział normalnych opinii na część testową i treningowa
    normal_opinions, normal_opinions_test = train_test_split(normal_opinions, test_size=config.get_value("test_size"))

    print("Ilość wykorzystanych opinii normalnych do treningu: ", len(normal_opinions))
    print("Ilość wykorzystanych opinii normalnych do testu: ", len(normal_opinions_test))
    print("Ilość wykorzystanych anomalii do testu: ", len(anomaly_opinions))
    return normal_opinions, normal_opinions_test


# Upraszczanie opinii przed dodaniem do DataSet (lower case, tylko litery)
def preprocess_review(review):
    # Usunięcie niepotrzebnych znaków
    cleaned_review = re.sub(config.get_value("regex"), '', review)
    cleaned_review = cleaned_review.lower()
    return cleaned_review


# Wyrównanie rozmiarów sekwencji wektorów
def pad_sequence(sequence, max_len):
    if sequence.size(0) == 0:
        return torch.zeros(max_len, )
    else:
        padded_seq = torch.zeros(max_len, sequence.size(1))
        padded_seq[:sequence.size(0)] = sequence
        return padded_seq


def create_tensors(normal_opinions, anomaly_opinions, normal_opinions_test):
    # Tworzenie datasetów dla danych normalnych treningowych, testowych i opinii o podwójnej jakości
    normal_dataset = OpinionDataset(normal_opinions)
    anomaly_dataset = OpinionDataset(anomaly_opinions)
    normal_dataset_test = OpinionDataset(normal_opinions_test)

    # Tworzenie tablic tensorów
    normal_dataset = [sequence for sequence in normal_dataset]
    anomaly_dataset = [sequence for sequence in anomaly_dataset]
    normal_dataset_test = [sequence for sequence in normal_dataset_test]
    return normal_dataset, normal_dataset_test, anomaly_dataset


def align_tensors_shape(normal_dataset, normal_dataset_test, anomaly_dataset):
    # Wyliczenie maksymalnego rozmiaru tensora
    max_seq_len1 = max(tensor.shape[0] for tensor in normal_dataset)
    print("Maksymalna długość tensora z zbioru treningowego: ", max_seq_len1)

    max_seq_len2 = max(tensor.shape[0] for tensor in normal_dataset_test)
    print("Maksymalna długość tensora z zbioru testowego: ", max_seq_len2)

    max_seq_len3 = max(tensor.shape[0] for tensor in anomaly_dataset)
    print("Maksymalna długość tensora z zbioru anomalii: ", max_seq_len3)

    max_seq_len = max(max_seq_len1, max_seq_len2, max_seq_len3)
    print("Maksymalna długość tensora: ", max_seq_len)

    # Dostosowanie tensorów do największego tensora za pomocą paddingu
    normal_dataset = [pad_sequence(sequence, max_seq_len) for sequence in normal_dataset]
    anomaly_dataset = [pad_sequence(sequence, max_seq_len) for sequence in anomaly_dataset]
    normal_dataset_test = [pad_sequence(sequence, max_seq_len) for sequence in normal_dataset_test]
    return normal_dataset, normal_dataset_test, anomaly_dataset


# Funkcja obliczająca próg na podstawie kwantyla
def calc_threshold(values, quantile):
    return torch.quantile(values, quantile)


# Funkcja trenująca autoenkodera
def train_autoencoder(model, dataloader, criterion, optimizer, quantile, cm, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = batch.float()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Obliczanie macierzy pomyłek
            with torch.no_grad():
                model.eval()
                mse_loss = nn.MSELoss(reduction='none')
                loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2))
                is_anomalous = torch.where(loss_values > calc_threshold(loss_values, quantile), 1, 0)

                for label in is_anomalous:
                    cm[label][label] += 1

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return calc_threshold(loss_values, quantile), cm


def start_training(normal_dataset):
    # Ilość cech tensora na wejściu
    input_dim = normal_dataset[0].shape[1]

    # Inicjalizacja modelu autoenkodera
    autoencoder = Autoencoder(input_dim)

    # Definicja funkcji straty i optymalizatora
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.get_value("lr"))

    # Inicjalizacja pustej macierzy pomyłek
    cm = torch.zeros((2, 2), dtype=torch.int)

    # Tworzenie DataLoader dla danych normalnych
    normal_dataloader = DataLoader(normal_dataset, batch_size=config.get_value("batch_size_train"), shuffle=True)

    # Ustalenie wartości kwantyla
    quantile = config.get_value("quantile")

    # Trenowanie autoenkodera
    threshold, cm = train_autoencoder(autoencoder, normal_dataloader, criterion, optimizer, quantile, cm, num_epochs=config.get_value("epochs"))

    print("Wyznaczony próg podczas treningu: ", threshold.item())
    print("")

    # Wyświetlanie macierzy pomyłek
    print("Macierz pomyłek z treningu: ")
    print(cm)
    return autoencoder, threshold


def save_model(autoencoder, threshold):
    # Zapis modelu"
    torch.save(autoencoder.state_dict(), "models/model.pth")

    # Zapis obliczonego progu
    config.save_threshold(threshold.item())

    print("Model i obliczony próg zapisany!")


def test_normal_opinions(autoencoder, normal_dataset_test, threshold):
    # Inicializacja pustej macierzy cm
    cm = torch.zeros((2, 2), dtype=torch.int)

    # Testowanie na opiniach nieświadczących o podwójnej jakości
    with torch.no_grad():
        autoencoder.eval()
        for batch in DataLoader(normal_dataset_test, batch_size=config.get_value("batch_size_test")):
            inputs = batch.float()
            outputs = autoencoder(inputs)
            mse_loss = nn.MSELoss(reduction='none')
            loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2))
            is_anomalous = torch.where(loss_values > threshold, 1, 0)
            cm[0][is_anomalous] += 1

    return cm


def test_anomaly_opinions(autoencoder, anomaly_dataset, threshold):
    # Inicializacja pustej macierzy cm
    cm = torch.zeros((2, 2), dtype=torch.int)
    counter = 1
    # Testowanie na opiniach świadczących o podwójnej jakości
    with torch.no_grad():
        autoencoder.eval()
        for batch in DataLoader(anomaly_dataset, batch_size=config.get_value("batch_size_test")):
            inputs = batch.float()
            outputs = autoencoder(inputs)
            mse_loss = nn.MSELoss(reduction='none')
            loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2))
            is_anomalous = torch.where(loss_values > threshold, 1, 0)
            print("Anomalia nr: ", counter)
            print("Czy model wskazał anomalie: ", is_anomalous.item() == 1)
            print("Otrzymany loss: ", loss_values.item())
            cm[1][is_anomalous] += 1
            counter += 1
    return cm


def show_results(cm1,cm2):
    cm = cm1 + cm2
    # Obliczenie sumy wierszy i kolumn macierzy pomyłek
    sum_rows = torch.sum(cm, dim=1)
    sum_cols = torch.sum(cm, dim=0)

    # Obliczenie liczby prawdziwie negatywnych, fałszywie negatywnych, fałszywie pozytywnych i prawdziwie pozytywnych
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    # Wyświetlenie macierzy pomyłek
    print("Macierz pomyłek uzyskana podczas testów:")
    print(cm)
    print()

    # Wyświetlenie innych metryk oceny
    print("Liczba prawdziwie negatywnych (TN):", tn.item())
    print("Liczba fałszywie negatywnych (FN):", fn.item())
    print("Liczba fałszywie pozytywnych (FP):", fp.item())
    print("Liczba prawdziwie pozytywnych (TP):", tp.item())
    print()

    accuracy = (tn + tp) / (tn + fn + fp + tp)
    print("Accuracy:", accuracy.item())

    precision = tp / (tp + fp)
    print("Precision:", precision.item())

    recall = tp / (tp + fn)
    print("Recall:", recall.item())

    f1_score = 2 * (precision * recall) / (precision + recall)
    print("F1-Score:", f1_score.item())



