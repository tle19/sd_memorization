import os
import pandas as pd

def preprocessing(dataset, output_path, num_ppl):
    dataset_path = os.path.join('/data/tyler/datasets', dataset) #change based on root directory
    output_path = os.path.join(output_path, 'prompts.csv')
    csv_path = find_file(dataset_path)

    if '.tsv' in csv_path:    
        csv_file = pd.read_table(csv_path, sep='\t')
    else:
        csv_file = pd.read_csv(csv_path)

    df = pd.DataFrame(csv_file)
    original_name = column_name(dataset)

    df = df.rename(columns={original_name: 'Name'})
    df = df[df['Name'].apply(is_english)]
    df = df[['Name']]
    size = df.shape[0]
    
    if num_ppl > size:
        num_ppl = size

    df = df.sample(num_ppl).sort_values('Name')
    df.to_csv(output_path, index=False)
    prompts = df['Name'].tolist()

    return prompts

def is_english(s):
    return isinstance(s, str) and s.isascii()

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
        'voxceleb': 'speakers',
        'everypolitician': 'name',
        'musicbrainz': 'artist_mb'
    }
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise FileNotFoundError(dataset, 'dataset not located')