import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from utils import euclidean_distance, manhattan_distance, cosine_similarity, fid_distance

def metric(func):
    metric_functions = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_similarity,
        "fid": fid_distance
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
    args = parser.parse_args()
    return args

args = parse_args()
model_id = args.model_id
eval_metric = args.metric
dataset = args.dataset

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

csv_path = os.path.join('output/', dataset, 'prompts.csv')
prompts_df = pd.read_csv(csv_path)
generated_prompts = prompts_df['Name'].tolist()

distances = []
for index, prompt in enumerate(generated_prompts):

    if prompts_df['is_human'][index]:
        path = os.path.join('output/', dataset)

        x = os.path.join(path, 'images1', prompt + '.png')
        y = os.path.join(path, 'images2', prompt + '.png')
        
        image_x = Image.open(x)
        image_y = Image.open(y)

        # Image Embeddings
        inputs_x = processor(images=image_x, return_tensors="pt")
        inputs_y = processor(images=image_y, return_tensors="pt")

        # Image Features
        with torch.no_grad():
            image_features1 = model.get_image_features(**inputs_x).numpy()
            image_features2 = model.get_image_features(**inputs_y).numpy()

        dist = eval_metric(image_features1, image_features2)
        distances.append(dist)

    else:
        dist = -1
        distances.append(dist)

    print(prompt + ':')
    print(dist)

prompts_df['Metric'] = distances
prompts_df.to_csv(csv_path)

# additional metrics
distances = [dist for dist in distances if dist != -1]

print('   \033[1m' + eval_metric.__name__ + '\033[0m')
print('  Avg:', np.mean(distances))
print('  Max:', np.max(distances))
print('  Min:', np.min(distances))