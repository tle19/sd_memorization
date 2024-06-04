import torch
import argparse
import os
import numpy as np
from diffusers import StableDiffusionPipeline
import pandas as pd
import shutil
from image_generation import generate_images
from preprocessing import imdb_preprocessing
from prompt_generation import generate_prompts


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
imdb_preprocessing()
tsv_file_path = os.path.join('/home/tyler/people_data/modified/', prompt_type + '.tsv')
prompts_df = pd.read_csv(tsv_file_path, sep='\t').sample(100) #sampling 10 prompts for easy computation
prompts = prompts_df.iloc[:, 0].values.tolist()

# Directory Initilization
output_path = os.path.join('output/', prompt_type)
if os.path.exists(output_path):
    shutil.rmtree(output_path)

sd_folder_path1 = os.path.join(output_path, 'images1')
sd_folder_path2 = os.path.join(output_path, 'images2')

os.makedirs(sd_folder_path1)
os.makedirs(sd_folder_path2)

print('Initialized', prompt_type, 'directory')

# Image and Prompt Generation
generate_images(model_id, prompts, prompts, sd_folder_path1)

csv_file_path = os.path.join(output_path, 'prompts.tsv')
prompts_df.to_csv(csv_file_path)


generate_prompts(model_id, prompts, sd_folder_path1, output_path)

generated_prompts_df = pd.read_csv(csv_file_path)
generated_prompts = generated_prompts_df.iloc[:, 2].values.tolist()

generate_images(model_id, prompts, generated_prompts, sd_folder_path2)

