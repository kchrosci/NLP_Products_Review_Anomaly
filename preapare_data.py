import pandas as pd
import re

def drop_unused_columns_anomaly(input_file, output_file):
    df = pd.read_csv(input_file, sep=',')
    #df = df.loc[df['language'] == "POL"]
    df = df.loc[df['doubleQuality'] == 1]
    selected_columns = ['content', 'doubleQuality']
    df = df.drop(df.columns.difference(selected_columns), axis=1)
    df['content'] = df['content'].str.lower().apply(lambda x: re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', '', x))
    #df = df.drop_duplicates(subset=['content'], keep='first')
    df.to_csv(output_file, index=False)

def drop_unused_columns_normal(input_file, output_file):
    df = pd.read_csv(input_file, sep=',')
    #df = df.loc[df['language'] == "POL"]
    selected_columns = ['content', 'doubleQuality']
    df = df.drop(df.columns.difference(selected_columns), axis=1)
    df['doubleQuality'] = df['doubleQuality'].replace({True: 1, False: 0})
    df['content'] = df['content'].str.lower().apply(lambda x: re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', '', x))
    #df = df.drop_duplicates(subset=['content'], keep='first')
    df = df[(df['content'].str.len() >= 2)]
    df.to_csv(output_file, index=False)


input_normal_file = "csv_data/full_dataset_translated/normal_opinions_full_dataset.csv"
input_anomaly_file = "csv_data/full_dataset_translated/anomaly_opinions_full_dataset.csv"
output_normal_file = "csv_data/preprocesed_files/normal_opinions2.csv"
output_anomaly_file = "csv_data/preprocesed_files/anomaly_opinions.csv"

drop_unused_columns_normal(input_normal_file, output_normal_file)
drop_unused_columns_anomaly(input_anomaly_file, output_anomaly_file)



