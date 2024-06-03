import torch
import argparse
import os
import numpy as np
from diffusers import StableDiffusionPipeline
import pandas as pd
import shutil
from utils import euclidean
from image_generation import generate_images
# from prompt_generation import generate_prompts


def parse_args():
    parser = argparse.ArgumentParser(description="SD Image Generation")
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--prompt', type=str, default="popular_actors")
    parser.add_argument('--metric', type=str, default="euclidean")
    args = parser.parse_args()
    return args

args = parse_args()

model_id = args.model_id
prompt_type = args.prompt

# Compiling Prompts
tsv_file_path = os.path.join('/home/tyler/people_data/modified/', prompt_type + '.tsv')
prompts_df = pd.read_csv(tsv_file_path, sep='\t').sample(10)
prompts = prompts_df.values.tolist()

# Directory Initilization
output_path = os.path.join('output/', prompt_type)
if os.path.exists(output_path):
    shutil.rmtree(output_path)

sd_folder_path = os.path.join(output_path, 'images')
blip_folder_path = os.path.join(output_path, 'annotations')
os.makedirs(sd_folder_path)
os.makedirs(blip_folder_path)
print('Initialized', prompt_type, 'directory')

# Image and Prompt Generation
generate_images(model_id, prompt_type, prompts)
# generate_prompts()

# metric calculation
for i in range(len(prompts)):
    x = 'output/popular_actors/images/00000001.png'
    y = 'output/popular_actors/images/00000002.png'
    euclidean(x, y)
