import os
import argparse
import pandas as pd
import shutil
from image_generation import *
from prompt_generation import *
from utils import imdb_preprocessing
    
def parse_args():
    parser = argparse.ArgumentParser(description="Image & Prompt Generation")
    parser.add_argument('--sd_model', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--blip_model', type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument('--dataset', type=str, default="popular_actors")
    args = parser.parse_args()
    return args

args = parse_args()
sd_id = args.sd_model
blip_id = args.blip_model
dataset = args.dataset

# Compiling Prompts
# dataset_path = preprocessing(dataset)
imdb_preprocessing()
dataset_path = os.path.join('/home/tyler/datasets/imdb/', dataset + '.csv')
prompts_df = pd.read_csv(dataset_path).sample(10) #sampling 10 prompts for easy computation
prompts = prompts_df['Name'].tolist()

# Directory Initilization
output_path = os.path.join('output/', dataset)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

csv_file_path = os.path.join(output_path, 'prompts.csv')
prompts_df.to_csv(csv_file_path)

sd_folder_path1 = os.path.join(output_path, 'images1')
sd_folder_path2 = os.path.join(output_path, 'images2')

os.makedirs(sd_folder_path1)
os.makedirs(sd_folder_path2)

print('Initialized', dataset, 'directory')

# Load SD & BLIP Models
sd_model(sd_id)
blip_model(blip_id)

# Image and Prompt Generation
generate_images(prompts, prompts, sd_folder_path1)

generated_prompts = generate_prompts(prompts, sd_folder_path1, output_path)

generate_images(prompts, generated_prompts, sd_folder_path2)

