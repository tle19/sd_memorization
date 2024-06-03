import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import pandas as pd
import os

def generate_images(model_id, prompt_type, prompts):
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)

    device = 'cuda'
    pipe = pipe.to(device)

    start_val = 0
    counter = '{:0{width}d}'.format(0, width=8)

    for prompt in prompts:
        print('IMAGE', counter, ':')
        image = pipe(prompt).images[0]  
        image_path = os.path.join('output/', prompt_type, 'images', counter + '.png')
        image.save(image_path)

        start_val += 1
        counter = '{:0{width}d}'.format(start_val, width=8)

