import os
import argparse
import pandas as pd
from image_generation import ImageGeneration
from caption_generation import CaptionGeneration
from preprocessing import preprocessing, save_csv
    
def parse_args():
    parser = argparse.ArgumentParser(description="Image & Prompt Generation")
    parser.add_argument('--sd_model', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--blip_model', type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--num_ppl', type=int, default=99999999)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--seed', type=int, default=42) #change to default=None later
    args = parser.parse_args()
    return args

args = parse_args()
sd_id = args.sd_model
blip_id = args.blip_model
dataset = args.dataset
num_ppl = args.num_ppl
one_prompt = args.prompt
seed = args.seed

# Dataset Preprocessing
if one_prompt == '':
    df = preprocessing(dataset, num_ppl, seed)
else:
    dataset = 'prompts'
    df = pd.DataFrame([one_prompt], columns=['Name']) 

# Directory Initilization
count = 0
output_path = os.path.join('output', dataset, f'{dataset}_{count}')

while os.path.exists(output_path):
    count += 1
    output_path = os.path.join('output', dataset, f'{dataset}_{count}')

os.makedirs(output_path)

image_path1 = os.path.join(output_path, 'images1')
image_path2 = os.path.join(output_path, 'images2')

os.makedirs(image_path1)
os.makedirs(image_path2)

save_csv(df, output_path)
prompts = df['Name'].tolist()

print(f'Initialized {dataset}_{count} directory')
print('Images to generate:', len(prompts))

# Load SD & BLIP Models
sd_model = ImageGeneration(sd_id, seed)
blip_model = CaptionGeneration(blip_id)

# Image and Prompt Generation
sd_model.generate_images(prompts, prompts, image_path1)

generated_prompts = blip_model.generate_captions(prompts, image_path1, output_path)

sd_model.generate_images(prompts, generated_prompts, image_path2)