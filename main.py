import time

import spacy
import numpy as np
import torch
import torch.nn as nn
from torch import tanh
from torch.utils.data import DataLoader, Dataset
from pandas import read_csv

#Początek pomiaru czasu
start = time.time()

d_true = read_csv('datasets/doubleQuality.csv', sep=',')
d_false = read_csv('datasets/nlp2023_v2.csv', sep=',')

# Przykładowe dane opinii (normalne i opinie o podwójnej jakości)
d_true = d_true.loc[d_true['language'] == "POL"]
d_false = d_false.loc[d_false['salusBcContent.language'] == "POL"]
d_false = d_false.loc[d_false['salusBcContent.doubleQuality'] == False]
anomalous_opinions = d_true["content"].values.tolist()
normal_opinions = d_false["salusBcContent.description"].values.tolist()

# Inicjalizacja modelu spaCy
nlp = spacy.load('pl_core_news_lg')

# CUDA
device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)


# Klasa Dataset do przechowywania danych opinii
class OpinionDataset(Dataset):
    def __init__(self, opinions):
        self.opinions = opinions

    def __len__(self):
        return len(self.opinions)

    def __getitem__(self, idx):
        doc = nlp(self.opinions[idx])
        sequence = np.array([token.vector for token in doc])
        sequence = torch.tensor(sequence, dtype=torch.float32)
        return sequence


# Tworzenie datasetów dla danych normalnych i opinii o podwójnej jakości
normal_dataset = OpinionDataset(normal_opinions)
anomalous_dataset = OpinionDataset(anomalous_opinions)

# Wyliczenie maksymalnego rozmiaru sekwencji wektorów
max_seq_len = max(len(opinion) for opinion in normal_dataset.opinions + anomalous_dataset.opinions)


# Wyrównanie rozmiarów sekwencji wektorów
def pad_sequence(sequence, max_len):
    padded_seq = torch.zeros(max_len, sequence.size(1))
    padded_seq[:sequence.size(0)] = sequence
    return padded_seq


# Dostosowanie sekwencji wektorów do maksymalnego rozmiaru
normal_dataset = [pad_sequence(sequence, max_seq_len) for sequence in normal_dataset]
anomalous_dataset = [pad_sequence(sequence, max_seq_len) for sequence in anomalous_dataset]


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
optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.005)


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

#Próg
threshold = 0

# Tworzenie DataLoader dla danych normalnych
normal_dataloader = DataLoader(normal_dataset, batch_size=32, shuffle=True)

# Trenowanie autoenkodera
threshold = train_autoencoder(autoencoder, normal_dataloader, criterion, optimizer, num_epochs=10)
print("Wyznaczony próg: ",threshold)

# Testowanie na danych opinii o podwójnej jakości
counter = 0
with torch.no_grad():
    autoencoder.eval()
    for batch in DataLoader(anomalous_dataset, batch_size=1):
        inputs = batch.float()
        outputs = autoencoder(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss_values = mse_loss(outputs, inputs).mean(dim=(1, 2)).numpy()
        print("Przewidziane wartości błędu dla opinii:", loss_values)
        is_anomalous = np.where(loss_values > threshold, 1, 0)
        if is_anomalous:
            counter = counter + 1
        print("Czy opinia jest oznaczona jako anomalia:", np.any(is_anomalous))

#Koniec pomiaru czasu
elapsed = time.time() - start

print("Wykryty procent anomalii: ",counter/len(anomalous_opinions))
print("Liczba opinii świadczących o podwójnej jakości: ",len(anomalous_opinions))
print("Liczba normalnych opinii: ",len(normal_opinions))
print("Czas trwania tokenizacji, szkolenia i testu: ", elapsed)
