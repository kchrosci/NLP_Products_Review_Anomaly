import torch
import torch.nn as nn
from config_loader import ConfigLoader
import spacy
import numpy as np

# Wczytywanie configu
config = ConfigLoader("config.json")
config.load_config()

nlp = spacy.load(config.get_value("spacy"))

# Definicja autoenkodera (bedzie w oddzielnym pliku później)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 6),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim // 6, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Tworzenie instancji modelu
model = Autoencoder(config.get_value("spacy_size"))

# Wczytanie zapisanych wag do modelu
model.load_state_dict(torch.load("models/model.pth"))

opinia = "To jest zajebisty produkt! Lepszy niż Niemiecki"
print("Testowana opinia: ")
print(opinia)

# Obróbka testowanej opinii na odpowiedni format
doc = nlp(opinia)
input_data = np.array([token.vector for token in doc])
input_data = torch.tensor(input_data, dtype=torch.float32)

# Wczytanie określonej podczas treningu wartości progu
threshold = torch.tensor(config.get_value("threshold"))

# Wykonanie predykcji na przykładowych danych
model.eval()
output = model(input_data)
mse_loss = nn.MSELoss(reduction='none')
loss_values = mse_loss(output, input_data).mean()
is_anomalous = torch.where(loss_values > threshold, 1, 0)
print("Czy model wskazał anomalie: ", is_anomalous.item() == 1)