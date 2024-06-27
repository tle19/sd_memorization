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
    print(f"SEED {seed}")
else:
    print("NO SEED")

if model_type == 'clip':
    model = CLIPEmbed(cuda)
elif model_type == 'dino':
    model = DINOEmbed(cuda)
else:
    raise TypeError('Embedding type not found')

output_path = os.path.join('output', dataset, file)
csv_path = os.path.join(output_path, 'prompts.csv')
base_images = os.path.join(output_path, 'base_images')

num_iters = folder_size(output_path)

prompts_df = pd.read_csv(csv_path)
names = prompts_df['Name'].tolist()

cosine_scores = []
fid_scores = []
isc_scores = []

for i in range(num_iters):
    generated_images = os.path.join(output_path, f'generated_images_{i}')

    fid_and_isc = calculate_fid(base_images, generated_images)
    isc_batch_score = fid_and_isc['inception_score_mean']
    fid_batch_score = fid_and_isc['frechet_inception_distance']
    cos_batch_scores = []

    print(f'\n\033[1m  BATCH {i}:\033[0m')

    for index, name in enumerate(names):
        print_title('IMAGE', name, index)
        if prompts_df['is_human'][index]:
            
            x = os.path.join(base_images, name + '.png')
            y = os.path.join(generated_images, name + '.png')

            features_x = model.image_feature(x)
            features_y = model.image_feature(y)
            
            score = cosine_similarity(features_x, features_y)[0, 0]
            cos_batch_scores.append(score)
        else:
            cos_batch_scores.append(-1)
        
    cosine_scores.append(cos_batch_scores)
    fid_scores.append(fid_batch_score)
    isc_scores.append(isc_batch_score)

prompts_df['Cosine'] = np.mean(cosine_scores, axis=0)
prompts_df['FID'] = np.mean(fid_scores)
prompts_df['IS'] = np.mean(isc_scores)

prompts_df.to_csv(csv_path, index=False)

# printed metrics
distances = prompts_df['Cosine'][prompts_df['is_human']]

print('\n\033[1mMetrics\033[0m')
print(f'Cosine Score: {np.mean(distances)}')
print(f'FID Score: {np.mean(fid_scores)}')
print(f'IS Score: {np.mean(isc_scores)}')