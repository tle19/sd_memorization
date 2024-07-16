import os
import pandas as pd
import re

def preprocessing(dataset, num_ppl, seed):
    dataset_path = os.path.join('/data/tyler/datasets', dataset) #change based on root directory
    csv_path = find_file(dataset_path)

    print('Processing Data...')
    if '.tsv' in csv_path:    
        csv_file = pd.read_csv(csv_path, sep='\t', na_values=['\\N', ''])
    else:
        csv_file = pd.read_csv(csv_path, na_values=['\\N', ''])

    df = pd.DataFrame(csv_file)
    original_name = column_name(dataset)

    df = df.rename(columns={original_name: 'Name'})
    df = df[['Name']]

    df = df.drop_duplicates(subset='Name', keep='first')
    df = df[df['Name'].apply(is_english)]
    df = df[df['Name'].apply(is_name)]
    df = df.dropna()

    size = df.shape[0]
    
    if num_ppl > size:
        num_ppl = size

    df = df.sample(num_ppl, random_state=seed).sort_values('Name')

    return df

def is_english(s):
    return isinstance(s, str) and s.isascii()

def is_name(text):
    name_pattern = r'^([A-Z][a-z]+(-[A-Z][a-z]+)?\s?|[A-Z]\.\s?)+[A-Z][a-z]+(-[A-Z][a-z]+)?$'
    return bool(re.match(name_pattern, text))

def find_file(dataset_path):
    files = os.listdir(dataset_path)

    if len(files) > 0:
        csv_file = files[0]
        csv_path = os.path.join(dataset_path, csv_file)
        return csv_path
    else:
        raise FileNotFoundError('No CSV file located')

def column_name(dataset):
    datasets = {
        'imdb': 'primaryName',
        'everypolitician': 'name'
        # 'dataset_name': 'column_name'
    }
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise FileNotFoundError(dataset, 'dataset not located')