import torch
import argparse
import os
import numpy as np
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="SD Image Generation")
    parser.add_argument('--model_id', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--metric', type=str, default="euclidean")
    parser.add_argument('--prompt', type=str, default="popular_actors")
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

# Device
model_id = args.model_id
device = 'cuda'

# Pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Prompts
prompt = 'a photo of an astronaut riding a horse on mars'
image = pipe(prompt).images[0]  
image = pipe(prompt)


image.save('output/.png')
