# NLP - Przetwarzanie języka naturalnego 23L
## NLP_Products_Review_Anomaly
### Temat projektu

Projekt polega na stworzeniu modelu wykrywania podwójnych opinii w zdaniach dotyczących produktów: proszków do prania kolorów, tabletek do zmywarki i kubków termicznych na podstawie dostarczonych danych. Jest to problem rozwiązany z podejsciem wykrywania anomalii. Język: polski.

### Zespół
* kchrosci
* Rysiek98

## Opis projektu
NLP_Products_Review_Anomaly jest projektem z zakresu przetwarzania języka naturalnego (NLP), którego celem jest opracowanie skutecznego systemu wykrywania podwójnych opinii w recenzjach produktów. W szczególności, skupiamy się na trzech kategoriach produktów: proszków do prania kolorów, tabletek do zmywarki i kubków termicznych.

## Opis projektu
Dane, na których pracujemy są dostarczone w formie zbioru recenzji produktów w języku polskim. W tym repozytorium pliki zostały zapisane w folderze /csv_data. Znajdują się oryginalne pliki .csv w folderze /original, częściowo przetworzone w folderach /auxilliary, /training_dataset, a także w pełni przetworzone wstępnie i wykorzystane w programach /preprocesed_files.

* all_opinions_merged - zawiera wszystkie opinie (podwójna jakość i normalne)
* anomaly_opinions - same podwójne opinie
* noraml_opinions - same normalne opinie

## Metody i techniki
* Zastosowano różne metody i techniki przetwarzania języka naturalnego, takie jak tokenizacja, usuwanie stopwords, stemming, wektoryzacja tekstu oraz algorytmy uczenia maszynowego, w tym model oparty na sieciach neuronowych (Autoencoder). 
* Skorzystano również z regułowego wykrywania podwójnych opinii, które pozwolają nam porównać działania modelu.

## Narzędzia
Do implementacji projektu użyjemy języka Python oraz popularnych bibliotek NLP, takich jak spaCy, czy PyTorch. Ponadto, wykorzystano narzędzia do wizualizacji danych, takie jak matplotlib i seaborn, aby przedstawić wyniki naszych analiz w czytelny sposób.

## Etapy projektu
1. Zebranie i przygotowanie danych - otrzymane dane zostaną przetworzone i poddane analizie w celu przygotowania ich do modelowania.

2. Przetwarzanie języka naturalnego - zastosowano techniki NLP, takie jak tokenizacja, usuwanie stopwords, stemming i wektoryzacja tekstu, aby przygotować dane tekstowe do uczenia maszynowego.

3. Modelowanie - model uczenia maszynowego, który jest w stanie wykrywać anomalie w opiniach produktowych.

4. Ewaluacja - oceniona jest skuteczność opracowanych modeli, korzystając z odpowiednich metryk i technik walidacji.

5. Wizualizacja wyników - przedstawiono wyniki naszych analiz za pomocą wykresów i diagramów, aby ułatwić zrozumienie zebranych informacji.

6. Dokumentacja i przygotowanie końcowego raportu - opisano nasze podejście, wyniki i wnioski w formie czytelnego raportu.

## Wymagania
Do uruchomienia projektu wymagane są następujące zależności:

Python 3.9 oraz nowszy
Biblioteki Python: spaCy, scikit-learn, PyTorch, matplotlib, seaborn

## Instrukcje uruchomienia
1. Sklonuj repozytorium projektu: git clone https://github.com/NLP_Products_Review_Anomaly.git

2. Przejdź do katalogu projektu: cd NLP_Products_Review_Anomaly

3. Zainstaluj wymagane biblioteki: 

4. Uruchom skrypt: python main.py

5. Podążąj za instrukcjami na ekranie.

## Autorzy
* kchrosci
* Rysiek98