import torch
import torch.nn as nn
from config_loader import ConfigLoader
from autoencoder import Autoencoder
import spacy
import numpy as np

# Wczytywanie configu
config = ConfigLoader("config.json")
config.load_config()

nlp = spacy.load(config.get_value("spacy"))

# Tworzenie instancji modelu
model = Autoencoder(config.get_value("spacy_size"))

# Wczytanie określonej podczas treningu wartości progu
threshold = torch.tensor(config.get_value("threshold"))

# Wczytanie zapisanych wag do modelu
model.load_state_dict(torch.load("models/model.pth"))

# Obróbka testowanej opinii na odpowiedni format
def preprocess_opinion(opinion):
    doc = nlp(opinion)
    input_data = np.array([token.vector for token in doc])
    input_data = torch.tensor(input_data, dtype=torch.float32)
    return input_data

# Wykonanie predykcji na przykładowych danych
def predict(input_data):
    model.eval()
    output = model(input_data)
    mse_loss = nn.MSELoss(reduction=config.get_value("reduction"))
    loss_values = mse_loss(output, input_data).mean()
    is_anomalous = torch.where(loss_values > threshold, 1, 0)
    print("Czy model wskazał anomalie: ", is_anomalous.item() == 1)

flag = True
print("Witaj!")

while(flag):
    opinion = input("Proszę podać opinię o produkcie do przetestowania przez model: ")
    input_data = preprocess_opinion(opinion)
    predict(input_data)
    what_next = input("Jeśli chcesz wprowadzić nową opinię wpisz t, jeśli chcesz zakończyć wpisz inny znak: ")
    if(what_next.lower() != 't'):
        flag = False
        print("Dziękuje za test! Do widzenia!")
