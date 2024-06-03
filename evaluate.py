import torch
import argparse
import os
import numpy as np
from diffusers import StableDiffusionPipeline
import pandas as pd
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="SD Image Generation")
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--prompt', type=str, default="popular_actors")
    parser.add_argument('--metric', type=str, default="euclidean")
    args = parser.parse_args()
    return args

args = parse_args()

# Device
model_id = args.model_id
device = 'cuda'

# Pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.safety_checker = lambda images, clip_input: (images, False)
pipe = pipe.to(device)

# Prompt Generation
prompt_type = args.prompt
tsv_file_path = os.path.join('/home/tyler/people_data/modified/', prompt_type + '.tsv')
prompts_df = pd.read_csv(tsv_file_path, sep='\t')
prompts = prompts_df.values.tolist()

output_path = os.path.join('output/', prompt_type)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
image_folder_path = os.path.join(output_path, 'images')
os.makedirs(image_folder_path)
print('Initialized', prompt_type, 'directory')

start_val = 0
counter = '{:0{width}d}'.format(0, width=8)
for prompt in prompts:
    image = pipe(prompt).images[0]  
    image_path = os.path.join('output/', prompt_type, 'images', counter + '.png')
    image.save(image_path)

    start_val += 1
    counter = '{:0{width}d}'.format(start_val, width=8)
