import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--file', type=str, default="imdb_0")
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42) #change to default=-1 later
    args = parser.parse_args()
    return args

args = parse_args()
model_type = args.model
file = args.file
cuda = args.cuda
seed = args.seed
dataset = punc_splice('_', file)

if model_type == 'clip':
    model = CLIPEmbed(seed, cuda)
elif model_type == 'dino':
    model = DINOEmbed(seed, cuda)
else:
    raise TypeError('Embedding type not found')

output_path = os.path.join('output', dataset, file)
csv_path = os.path.join(output_path, 'prompts.csv')
prompts_df = pd.read_csv(csv_path)
generated_prompts = prompts_df['Name'].tolist()

distances = []

for index, prompt in enumerate(generated_prompts):

    if prompts_df['is_human'][index]:
        x = os.path.join(output_path, 'images1', prompt + '.png')
        y = os.path.join(output_path, 'images2', prompt + '.png')

        features_x = model.image_feature(x)
        features_y = model.image_feature(y)

        dist = cosine_similarity(features_x, features_y)[0, 0]
    else:
        dist = -1

    distances.append(dist)

    print_title('IMAGE', prompt, index)
    print(dist)

prompts_df['Metric'] = distances
prompts_df.to_csv(csv_path, index=False)

# additional metrics
distances = [dist for dist in distances if dist != -1]

print('   \033[1m' + 'Cosine' + '\033[0m')
print('  Avg:', np.mean(distances))
print('  Max:', np.max(distances))
print('  Min:', np.min(distances))