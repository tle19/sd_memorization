from PIL import Image
import pandas as pd
import numpy as np

def open_image(path):
    return Image.open(path)

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def cosine_similarity(x, y):
    return (np.dot(x, y.T) / (euclidean_distance(x, 0) * euclidean_distance(y, 0)))[0][0]

#implement FID?

def convert_file(tsv_file):

    df = pd.read_table(tsv_file, sep='\t')
    csv_file = tsv_file.replace('.tsv', '.csv')
    df.to_csv(csv_file, index=False)

def imdb_preprocessing():

    tsv_file = '/home/tyler/datasets/imdb/name.basics.tsv'
    convert_file(tsv_file)
    csv_file = '/home/tyler/datasets/imdb/name.basics.csv'

    imdb = pd.read_csv(csv_file)
    imdb = pd.DataFrame(imdb)
    imdb = imdb.rename(columns={'primaryName': 'Name'})
    pop_actors = imdb[['Name']]

    print('Number of Popular Actors: ', pop_actors.shape[0])

    new_path = '/home/tyler/datasets/imdb/popular_actors.csv'
    pop_actors.to_csv(new_path, index=False)