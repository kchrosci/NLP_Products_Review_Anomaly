import pandas as pd
import re

def remove_duplicates_and_save(input_file, output_file):
    # Wczytanie danych z pliku CSV z odpowiednimi ustawieniami
    df = pd.read_csv(input_file, sep=',', quotechar='"', lineterminator='\n')

    # Połączenie opinii, które zajmują dwie linie
    df['content'] = df['content'].str.cat(df.groupby(df['content'].index // 2)['content'].transform(lambda x: ' '.join(x)))

    # Usunięcie duplikatów na podstawie przetworzonej kolumny "content"
    df['content_processed'] = df['content'].str.lower().apply(lambda x: re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', '', x))
    df = df.drop_duplicates(subset='content_processed')

    # Usunięcie dodatkowej kolumny
    df = df.drop(columns='content_processed')
    df['content'] = df['content'].str.lower().apply(lambda x: re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', '', x))
    # Usunięcie linii o zawartości "
    df = df[df['content'] != '"']

    # Usunięcie znaków nowej linii i spacji na początku i końcu każdej linii
    df['content'] = df['content'].str.replace('\n', ' ')
    df['content'] = df['content'].str.replace('\s+', ' ').str.strip()

    # Pominięcie linii krótszych niż 2 znaki
    df = df[df['content'].str.len() >= 2]

    # Zapis przetworzonego pliku CSV
    df.to_csv(output_file, index=False)

    print("Usunięto duplikaty, przetworzono tekst i zapisano przetworzony plik:", output_file)


input_file = "csv_data/training_dataset/normal_opinions.csv"
output_file = "csv_data/preprocesed_files/normal_opinions.csv"

remove_duplicates_and_save(input_file, output_file)