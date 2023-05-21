import re
import spacy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pandas import read_csv

# Wczytanie danych
anomaly_opinions = read_csv('csv_data/anomaly_opinions.csv', sep=',')
normal_opinions = read_csv('csv_data/normal_opinions.csv', sep=',')

anomaly_opinions = anomaly_opinions["content"].values.tolist()

normal_opinions = normal_opinions["content"].values.tolist()
normal_opinions = normal_opinions[:1000]
normal_opinions = normal_opinions[:len(normal_opinions) - 100]

normal_opinions_test = normal_opinions[len(normal_opinions) - 100:]

# Inicjalizacja modelu spaCy
nlp = spacy.load('pl_core_news_md')

# Inicjalizacja pustej macierzy pomyłek
cm = torch.zeros((2, 2), dtype=torch.int)

def preprocess_review(review):
    # Usunięcie niepotrzebnych znaków
    cleaned_review = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', '', review)
    # Tokenizacja tekstu
    cleaned_review = cleaned_review.lower()
    return nlp(cleaned_review)


# Klasa Dataset do przechowywania danych opinii
class OpinionDataset(Dataset):
    def __init__(self, opinions):
        self.opinions = opinions

    def __len__(self):
        return len(self.opinions)

    def __getitem__(self, idx):
        doc = preprocess_review(self.opinions[idx])
        sequence = np.array([token.vector for token in doc])
        sequence = torch.tensor(sequence, dtype=torch.float32)
        return sequence


# Tworzenie datasetów dla danych normalnych i opinii o podwójnej jakości

normal_dataset = OpinionDataset(normal_opinions)
anomalous_dataset = OpinionDataset(anomaly_opinions)
normal_test_dataset = OpinionDataset(normal_opinions_test)

# Wyliczenie maksymalnego rozmiaru sekwencji wektorów
max_seq_len1 = max(len(opinion) for opinion in normal_dataset.opinions)
max_seq_len2 = max(len(opinion) for opinion in anomalous_dataset.opinions)
max_seq_len3 = max(len(opinion) for opinion in normal_test_dataset.opinions)
max_seq_len = max(max_seq_len1, max_seq_len2, max_seq_len3)

# Wyrównanie rozmiarów sekwencji wektorów
def pad_sequence(sequence, max_len):
    if sequence.size(0) == 0:
        return torch.zeros(max_len, 300)
    else:
        padded_seq = torch.zeros(max_len, sequence.size(1))
        padded_seq[:sequence.size(0)] = sequence
        return padded_seq



# Dostosowanie sekwencji wektorów do maksymalnego rozmiaru
normal_dataset = [pad_sequence(sequence, max_seq_len) for sequence in normal_dataset]
anomalous_dataset = [pad_sequence(sequence, max_seq_len) for sequence in anomalous_dataset]
normal_dataset_test = [pad_sequence(sequence, max_seq_len) for sequence in normal_test_dataset]


# Definicja autoenkodera
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.Tanh(),
            nn.Linear(input_dim//2, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim//2),
            nn.Tanh(),
            nn.Linear(input_dim//2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return decoded


# Parametry modelu
input_dim = normal_dataset[0].size(1)

# Inicjalizacja modelu autoenkodera
autoencoder = Autoencoder(input_dim)

# Definicja funkcji straty i optymalizatora
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)


# Funkcja trenująca autoenkoder
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    loss_values = []
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
            loss_values.append(loss.item())
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Obliczenie progu
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    threshold = mean_loss + 2 * std_loss
    return threshold


# Próg
threshold = 0

# Tworzenie DataLoader dla danych normalnych
normal_dataloader = DataLoader(normal_dataset, batch_size=32, shuffle=True)

# Trenowanie autoenkodera
threshold = train_autoencoder(autoencoder, normal_dataloader, criterion, optimizer, num_epochs=10)
print("Wyznaczony próg: ", threshold)

counter = 0
with torch.no_grad():
    autoencoder.eval()
    for batch in DataLoader(normal_dataset_test, batch_size=1):
        inputs = batch.float()
        outputs = autoencoder(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
        is_anomalous = torch.tensor(np.where(loss_values > threshold, 1, 0))
        counter += torch.sum(is_anomalous).item()
        cm[0][is_anomalous] += 1

# Testowanie na danych opinii o podwójnej jakości
with torch.no_grad():
    autoencoder.eval()
    for batch in DataLoader(anomalous_dataset, batch_size=1):
        inputs = batch.float()
        outputs = autoencoder(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
        is_anomalous = torch.tensor(np.where(loss_values > threshold, 1, 0))
        counter += torch.sum(is_anomalous).item()
        cm[1][is_anomalous] += 1

# Obliczenie sumy wierszy i kolumn macierzy pomyłek
sum_rows = torch.sum(cm, dim=1)
sum_cols = torch.sum(cm, dim=0)

# Obliczenie liczby prawdziwie negatywnych, fałszywie negatywnych, fałszywie pozytywnych i prawdziwie pozytywnych
tn = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tp = cm[1][1]

# Wyświetlenie macierzy pomyłek
print("Macierz pomyłek:")
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






# # Testowanie na danych opinii o podwójnej jakości
# print("TEST NA OPINIACH ŚWIADCZĄCYCH O PODWÓJNEJ JAKOŚCI")
# counter = 0
# total_anomalies = 0
# with torch.no_grad():
#     autoencoder.eval()
#     for batch in DataLoader(anomalous_dataset, batch_size=1):
#         inputs = batch.float()
#         outputs = autoencoder(inputs)
#         mse_loss = nn.MSELoss(reduction='none')
#         loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
#         is_anomalous = np.where(loss_values > threshold, 1, 0)
#         counter += np.sum(is_anomalous)
#         total_anomalies += len(is_anomalous)
#
# # Obliczanie metryk
# recall = counter / total_anomalies
# precision = counter / len(anomaly_opinions)
# f1_score = 2 * (precision * recall) / (precision + recall)
#
# print("Czułość (recall): ", recall)
# print("Precyzja (precision): ", precision)
# print("F1-score: ", f1_score)
# print("Liczba opinii świadczących o podwójnej jakości: ", len(anomaly_opinions))
# print("Liczba normalnych opinii wykorzystanych do treningu: ", len(normal_opinions))
# print("                                                     ")
#
#
# print("TEST NA NORMALNYCH OPINIACH")
# counter = 0
# total_anomalies = 0
# with torch.no_grad():
#     autoencoder.eval()
#     for batch in DataLoader(normal_test_dataset, batch_size=1):
#         inputs = batch.float()
#         outputs = autoencoder(inputs)
#         mse_loss = nn.MSELoss(reduction='none')
#         loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
#         is_anomalous = np.where(loss_values > threshold, 1, 0)
#         counter += np.sum(is_anomalous)
#         total_anomalies += len(is_anomalous)
#
# # Obliczanie metryk
# recall = counter / total_anomalies
# precision = counter / len(anomaly_opinions)
# f1_score = 2 * (precision * recall) / (precision + recall)
#
# print("Czułość (recall): ", recall)
# print("Precyzja (precision): ", precision)
# print("F1-score: ", f1_score)
# print("Liczba opinii świadczących o podwójnej jakości: ", len(anomaly_opinions))
# print("Liczba normalnych opinii wykorzystanych do testu: ", len(normal_dataset_test))

# Testowanie na normalnych danych opinii