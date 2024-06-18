import os
import pandas as pd
import numpy as np

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def cosine_similarity(x, y):
    return (np.dot(x, y.T) / (euclidean_distance(x, 0) * euclidean_distance(y, 0)))[0][0]

def fid_distance(x, y):
    pass
    #implement FID

def preprocessing(dataset):
    dataset_path = os.path.join('/data/tyler/datasets/' + dataset) #change based on root directory
    csv_path = find_file(dataset_path)
    target = dataset_convert(dataset)

    new_path = os.path.join(dataset_path, target + '.csv')
    if os.path.exists(new_path):
        return new_path
    
    if '.tsv' in csv_path:    
        csv_file = pd.read_table(csv_path, sep='\t')
    else:
        csv_file = pd.read_csv(csv_path)

    df = pd.DataFrame(csv_file)
    original_name = column_name(dataset)
    df = df.rename(columns={original_name: 'Name'})
    df = df[['Name']]

    print('Number of ' + target + ': ', df.shape[0])
    
    df.to_csv(new_path, index=False)
    return new_path

def find_file(dataset_path):
    files = os.listdir(dataset_path)

    if len(files) > 0:
        csv_file = files[0]
        csv_path = os.path.join(dataset_path, csv_file)
        return csv_path
    else:
        raise ValueError('No CSV file located')

def dataset_convert(dataset):
    # IMDB, VoxCeleb, EveryPolitican, MusicBrainz
    datasets = {
        'imdb': 'actors',
        'voxceleb': 'celebrities',
        'everypolitican': 'politicians',
        'musicbrainz': 'musical artists'
    }
    if dataset in datasets:
        return datasets[dataset]
    else:
        raise ValueError("Invalid dataset provided")

def column_name(dataset):
    datasets = {
        'imdb': 'primaryName',
        'voxceleb': 'speakers',
        'everypolitican': 'person',
        'musicbrainz': 'artist_mb'
    }
    if dataset in datasets:
        return datasets[dataset]
    
def generate_graph():
    pass