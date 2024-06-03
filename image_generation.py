import torch
from diffusers import StableDiffusionPipeline
import os

def generate_images(model_id, prompts, sd_folder_path1):
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)

    device = 'cuda'
    pipe = pipe.to(device)

    start_val = 0
    counter = '{:0{width}d}'.format(0, width=8)

    for prompt in prompts:
        print('IMAGE', counter, ':')
        image = pipe(prompt).images[0]  
        image_path = os.path.join(sd_folder_path1, counter + '.png')
        image.save(image_path)

        start_val += 1
        counter = '{:0{width}d}'.format(start_val, width=8)

