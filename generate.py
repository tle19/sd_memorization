import os
import argparse
import pandas as pd
from transformers import set_seed
from image_generation import ImageGeneration
# from caption_generation import CaptionGeneration
from cogvlm import CaptionGeneration2
from preprocessing import preprocessing
    
def parse_args():
    parser = argparse.ArgumentParser(description="Image & Prompt Generation")
    parser.add_argument('--sd_model', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--blip_model', type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--input', type=int, default=99999999)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42) #change to default=None later
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset
num_steps = args.num_steps
one_prompt = args.prompt
cuda = args.cuda
seed = args.seed

# Optional Seed
if seed:
    set_seed(seed)
    print(f"SEED {seed}")
else:
    print("NO SEED")

# Dataset Preprocessing
if one_prompt == '':
    df = preprocessing(dataset, args.input, seed)
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
base_images = os.path.join(output_path, 'base_images')
os.makedirs(base_images)

prompts_csv = os.path.join(output_path, 'prompts.csv')
df.to_csv(prompts_csv, index=False)
names = df['Name'].tolist()

print(f'Directory {dataset}_{count} Initialized')
print(f'Base Images to Generate: {len(names)}')
print(f'Batch Images to Generate: {len(names)} x {args.batch}')

# Load SD & BLIP Models
sd_model = ImageGeneration(args.sd_model, num_steps, cuda)
# blip_model = CaptionGeneration(args.blip_model, args.temp, args.top_k, args.top_p, args.num_beams, cuda)
cogvlm_model = CaptionGeneration2(args.blip_model, args.temp, args.top_k, args.top_p, args.num_beams, cuda)

# Image and Prompt Generation
sd_model.generate_images(names, names, base_images)

# generated_captions = blip_model.generate_captions(names, base_images, output_path)
generated_captions = cogvlm_model.generate_captions(names, base_images, output_path)

for i in range(args.batch):
    print(f'\n\033[1m  BATCH {i}:\033[0m')

    generated_images = os.path.join(output_path, f'generated_images_{i}')
    os.makedirs(generated_images)

    sd_model.generate_images(names, generated_captions, generated_images)