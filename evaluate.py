import os
import argparse
import pandas as pd
import numpy as np
from transformers import set_seed
from sklearn.metrics.pairwise import cosine_similarity
from embedding import CLIPEmbed, DINOEmbed
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--file', type=str, default="imdb_0")
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42) #change to default=None later
    args = parser.parse_args()
    return args

args = parse_args()
model_type = args.model
file = args.file
cuda = args.cuda
seed = args.seed
dataset = punc_splice('_', file)

if seed:
    set_seed(seed)
    print(f"Seed {seed} set")
else:
    print("No seed set")

if model_type == 'clip':
    model = CLIPEmbed(cuda)
elif model_type == 'dino':
    model = DINOEmbed(cuda)
else:
    raise TypeError('Embedding type not found')

output_path = os.path.join('output', dataset, file)
csv_path = os.path.join(output_path, 'prompts.csv')
base_images = os.path.join(output_path, 'images1')
generated_images = os.path.join(output_path, 'images2')

prompts_df = pd.read_csv(csv_path)
names = prompts_df['Name'].tolist()

fid_scores = calculate_fid(base_images, generated_images)
cosine_scores = []

for index, name in enumerate(names):

    if prompts_df['is_human'][index]:
        x = os.path.join(base_images, name + '.png')
        y = os.path.join(generated_images, name + '.png')

        features_x = model.image_feature(x)
        features_y = model.image_feature(y)

        cos_score = cosine_similarity(features_x, features_y)[0, 0]
    else:
        cos_score = -1

    cosine_scores.append(cos_score)

    print_title('IMAGE', name, index)
    print(cos_score)

is_mean = fid_scores['inception_score_mean']
is_std = fid_scores['inception_score_std']
fid_score = fid_scores['frechet_inception_distance']

prompts_df['Cosine'] = cosine_scores
prompts_df['IS'] = is_mean
prompts_df['FID'] = fid_scores

prompts_df.to_csv(csv_path, index=False)

# printed metrics
distances = [dist for dist in cosine_scores if dist != -1]

print('\n\033[1mMetrics\033[0m')
print(f'Cosine Score: {np.mean(distances)}  \u00B1 {np.std(distances)}')
print(f'IS Score: {is_mean} \u00B1 {is_std}')
print(f'FID Score: {fid_score}')