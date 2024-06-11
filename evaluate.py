import argparse
import os
import torch
import pandas as pd
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from utils import *

def metric(func):
    metric_functions = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "cosine": cosine_similarity
    }
    
    if func in metric_functions:
        return metric_functions[func]
    else:
        raise argparse.ArgumentTypeError("Invalid metric provided")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--metric', type=metric, default="euclidean")
    parser.add_argument('--prompt', type=str, default="popular_actors")
    parser.add_argument('--model_id', type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    return args

args = parse_args()

eval_metric = args.metric
prompt_type = args.prompt
model_id = args.model_id

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

csv_path = os.path.join('output/', prompt_type, 'prompts.csv')
prompts_df = pd.read_csv(csv_path)
generated_prompts = prompts_df['Name'].tolist()

distances = []
for prompt in generated_prompts:
    path = os.path.join('output/', prompt_type)
    
    x = os.path.join(path, 'images1', prompt + '.png')
    y = os.path.join(path, 'images2', prompt + '.png')
    
    image_x = open_image(x)
    image_y = open_image(y)

    inputs_x = processor(images=image_x, return_tensors="pt")
    inputs_y = processor(images=image_y, return_tensors="pt")
    
    with torch.no_grad():
        image_features1 = model.get_image_features(**inputs_x).numpy()
        image_features2 = model.get_image_features(**inputs_y).numpy()

    dist = eval_metric(image_features1, image_features2)
    distances.append(dist)

    print(prompt, ':')
    print(dist)

# additional metrics
print('    \033[1m' + eval_metric.__name__ + '\033[0m')
print('Avg:', np.mean(distances))
print('Max:', np.max(distances))
print('Min:', np.min(distances))

prompts_df['Metric'] = distances
prompts_df.to_csv(csv_path)