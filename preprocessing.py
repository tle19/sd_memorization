import pandas as pd
import os

tsv_file_path = '/home/tyler/people_data/original/name.basics.tsv'
imdb = pd.read_csv(tsv_file_path, sep='\t')
pop_actors = imdb['primaryName'][1:]

print(pop_actors.head())
print(pop_actors.shape[0])

new_tsv_file_path = '/home/tyler/people_data/modified/popular_actors.tsv'
pop_actors.to_csv(new_tsv_file_path, sep='\t', index=False)
