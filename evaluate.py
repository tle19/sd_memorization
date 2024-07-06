import os
import argparse
import pandas as pd
import numpy as np
import shutil
import ast
from transformers import set_seed
from sklearn.metrics.pairwise import cosine_similarity
from embedding import CLIPEmbed, DINOEmbed
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--folder', type=str, default="imdb_0")
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42) #change to default=None later
    args = parser.parse_args()
    return args

args = parse_args()
model_type = args.model
folder = args.folder
cuda = args.cuda
seed = args.seed
dataset = punc_splice('_', folder)

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

output_path = os.path.join('output', dataset, folder)
csv_path = os.path.join(output_path, 'prompts.csv')
base_images = os.path.join(output_path, 'base_images')

num_iters = folder_size(output_path)

prompts_df = pd.read_csv(csv_path)
names = prompts_df['Name'].tolist()

cond = prompts_df['Name'][prompts_df['is_human']]
temp_base = os.path.join(output_path, 'temp_base')
make_temp_dir(base_images, temp_base, cond)

cosine_scores = []
fid_scores = []
isc_scores = []

for i in range(num_iters):
    generated_images = os.path.join(output_path, f'generated_images_{i}')

    temp_gen = os.path.join(output_path, 'temp_gen')
    make_temp_dir(generated_images, temp_gen, cond)

    fid_and_isc = calculate_fid(temp_base, temp_gen)
    isc_batch_scores = fid_and_isc['inception_score_mean']
    fid_batch_scores = fid_and_isc['frechet_inception_distance']
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
    
    shutil.rmtree(temp_gen)

    cosine_scores.append(cos_batch_scores)
    fid_scores.append(fid_batch_scores)
    isc_scores.append(isc_batch_scores)

shutil.rmtree(temp_base)

cosine_scores = np.array(cosine_scores).T
prompts_df['Cosine Avg'] = np.mean(cosine_scores, axis=1)
for i, score in enumerate(cosine_scores):
    prompts_df.loc[i, 'Cosine'] = str(score)
prompts_df['FID Avg'] = np.mean(fid_scores)
prompts_df['FID'] = str(fid_scores)
prompts_df['IS Avg'] = np.mean(isc_scores)
prompts_df['IS'] = str(isc_scores)

prompts_df.to_csv(csv_path, index=False)

# printed metrics
cosine_avg = prompts_df['Cosine Avg'][prompts_df['is_human']]

print('\n\033[1mMetrics\033[0m')
print(f'Cosine Score: {np.mean(cosine_avg)}')
print(f'FID Score: {np.mean(fid_scores)}')
print(f'IS Score: {np.mean(isc_scores)}')