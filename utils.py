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

def imdb_preprocessing():

    tsv_file = '/home/tyler/datasets/imdb/name.basics.tsv'
    tsv_to_csv(tsv_file)
    csv_file = '/home/tyler/datasets/imdb/name.basics.csv'

    imdb = pd.read_csv(csv_file)
    imdb = pd.DataFrame(imdb)
    imdb = imdb.rename(columns={'primaryName': 'Name'})
    pop_actors = imdb[['Name']]

    print('Number of Actors: ', pop_actors.shape[0])
    
    new_path = '/home/tyler/datasets/imdb/popular_actors.csv'
    pop_actors.to_csv(new_path, index=False)

# VoxCeleb, EveryPolitican, MusicBrainz datasets

def preprocessing(dataset):
    dataset_path = os.path.join('/home/tyler/datasets/' + dataset)
    csv_path = find_file(dataset_path)
    target = dataset_convert(dataset)
    if csv_path.contains('.tsv'):
        csv_path = tsv_to_csv()

    csv_file = pd.read_csv(csv_path)
    df = pd.DataFrame(csv_file)
    original_name = column_name(dataset)
    df = df.rename(columns={original_name: 'Name'})
    df = df[['Name']]

    print('Number of ' + target + ': ', df.shape[0])

    new_path = os.path.join(dataset_path, target + '.csv')
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

def tsv_to_csv(tsv_file):
    df = pd.read_table(tsv_file, sep='\t')
    csv_file = tsv_file.replace('.tsv', '.csv')
    df.to_csv(csv_file, index=False)
    return csv_file

def column_name(dataset):
    datasets = {
        'imdb': 'primaryName',
        'voxceleb': 'speakers',
        'everypolitican': 'person',
        'musicbrainz': 'artist_mb'
    }
    if dataset in datasets:
        return dataset