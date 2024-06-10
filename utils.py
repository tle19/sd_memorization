import numpy as np
from PIL import Image
import pandas as pd


def euclidean_func(x, y):
    image1 = Image.open(x)
    image2 = Image.open(y)

    vec1 = np.array(image1).flatten()
    vec2 = np.array(image2).flatten()

    distance = np.linalg.norm(vec1 - vec2)

    return distance

def manhattan_func(x, y):
    image1 = Image.open(x)
    image2 = Image.open(y)

    vec1 = np.array(image1).flatten()
    vec2 = np.array(image2).flatten()

    distance = np.linalg.norm(vec1 - vec2)

    return distance

#implement cosine, FID, manhattan, etc.

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