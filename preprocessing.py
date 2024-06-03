import pandas as pd
import os


# Open IMDB dataset
tsv_file_path = '/home/tyler/people_data/original/name.basics.tsv'
imdb = pd.read_csv(tsv_file_path, sep='\t')
pop_actors = imdb['primaryName']

# print(pop_actors.head()) #sanity check
print('Number of Popular Actors: ', pop_actors.shape[0])

# Save preprocessed file
new_tsv_file_path = '/home/tyler/people_data/modified/popular_actors.tsv'
if new_tsv_file_path.is_file():
    pop_actors.to_csv(new_tsv_file_path, sep='\t', index=False)
