import re
import spacy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pandas import read_csv

# Wczytanie danych
anomaly_opinions = read_csv('datasets/anomaly_opinions.csv', sep=';', encoding="cp1250")
normal_opinions = read_csv('datasets/normal_opinions.csv', sep=';', encoding="cp1250")

anomaly_opinions = anomaly_opinions["content"].values.tolist()

normal_opinions = normal_opinions["content"].values.tolist()
normal_opinions = normal_opinions[:1000]
normal_opinions = normal_opinions[:len(normal_opinions) - 100]

normal_opinions_test = normal_opinions[len(normal_opinions) - 100:]

# Inicjalizacja modelu spaCy
nlp = spacy.load('pl_core_news_md')


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
        self.fc1 = nn.Linear(input_dim, int(input_dim / 2))
        self.fc2 = nn.Linear(int(input_dim / 2), int(input_dim / 4))
        self.fc3 = nn.Linear(int(input_dim / 4), int(input_dim / 2))
        self.fc4 = nn.Linear(int(input_dim / 2), input_dim)

    def encode(self, x):
        z = torch.tanh(self.fc1(x))
        z = torch.tanh(self.fc2(z))  # latent in [-1,+1]
        return z

    def decode(self, x):
        z = torch.tanh(self.fc3(x))
        z = torch.sigmoid(self.fc4(z))  # [0.0, 1.0]
        return z

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z  # in [0.0, 1.0]


# Parametry modelu
input_dim = normal_dataset[0].size(1)

# Inicjalizacja modelu autoenkodera
autoencoder = Autoencoder(input_dim)

# Definicja funkcji straty i optymalizatora
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)


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

# Testowanie na danych opinii o podwójnej jakości
print("TEST NA OPINIACH ŚWIADCZĄCYCH O PODWÓJNEJ JAKOŚCI")
counter = 0
with torch.no_grad():
    autoencoder.eval()
    for batch in DataLoader(anomalous_dataset, batch_size=1):
        inputs = batch.float()
        outputs = autoencoder(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
        # print("Przewidziane wartości błędu dla opinii:", loss_values)
        is_anomalous = np.where(loss_values > threshold, 1, 0)
        if is_anomalous:
            counter = counter + 1
        # print("Czy opinia jest oznaczona jako anomalia:", np.any(is_anomalous))

print("Wykryty procent anomalii: ", counter / len(anomaly_opinions))
print("Liczba opinii świadczących o podwójnej jakości: ", len(anomaly_opinions))
print("Liczba normalnych opinii wykorzystanych do treningu: ", len(normal_opinions))

print("TEST NA NORMALNYCH OPINIACH")
counter = 0
with torch.no_grad():
    autoencoder.eval()
    for batch in DataLoader(normal_dataset_test, batch_size=1):
        inputs = batch.float()
        outputs = autoencoder(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
        # print("Przewidziane wartości błędu dla opinii:", loss_values)
        is_anomalous = np.where(loss_values > threshold, 1, 0)
        if is_anomalous:
            counter = counter + 1
        # print("Czy opinia jest oznaczona jako anomalia:", np.any(is_anomalous))

print("Procent błędnie wykrytych anomalii: ", counter / len(normal_dataset_test))
print("Liczba normalnych opinii wykorzystanych do testu: ", len(normal_dataset_test))
