import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import set_seed
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from utils import *

def metric(func):
    metric_functions = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_similarity,
        "fid": fid_distance,
        "is": is_distance
    }
    
    if func in metric_functions:
        return metric_functions[func]
    else:
        raise argparse.ArgumentTypeError("Invalid metric provided")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--model_id', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument('--metric', type=metric, default="cosine")
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--folder', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

args = parse_args()
model_id = args.model_id
eval_metric = args.metric
dataset = args.dataset
folder = args.folder

set_seed(args.seed)

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

output_path = os.path.join('output', dataset, dataset + '_' + str(folder))
csv_path = os.path.join(output_path, 'prompts.csv')
prompts_df = pd.read_csv(csv_path)
generated_prompts = prompts_df['Name'].tolist()

distances = []

def image_feature(z):
    image = Image.open(z)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_feature = model.get_image_features(**inputs).numpy()

    return image_feature

for index, prompt in enumerate(generated_prompts):

    if prompts_df['is_human'][index]:
        x = os.path.join(output_path, 'images1', prompt + '.png')
        y = os.path.join(output_path, 'images2', prompt + '.png')

        dist = eval_metric(image_feature(x), image_feature(y))
    else:
        dist = -1

    distances.append(dist)

    print_title('IMAGE', prompt, index)
    print(dist)

prompts_df['Metric'] = distances
prompts_df.to_csv(csv_path, index=False)

# additional metrics
distances = [dist for dist in distances if dist != -1]

print('   \033[1m' + eval_metric.__name__ + '\033[0m')
print('  Avg:', np.mean(distances))
print('  Max:', np.max(distances))
print('  Min:', np.min(distances))