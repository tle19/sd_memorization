import torch
from diffusers import StableDiffusionPipeline
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None

def sd_model(model_id):
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)
    pipe = pipe.to(device)

def generate_images(names, prompts, sd_folder_path1):
    
    start_val = 0
    counter = '{:0{width}d}'.format(0, width=8)

    for prompt in prompts:
        print('IMAGE', counter, '-', names[start_val])

        image = pipe(prompt).images[0]  
        image_path = os.path.join(sd_folder_path1, names[start_val] + '.png')
        image.save(image_path)

        start_val += 1
        counter = '{:0{width}d}'.format(start_val, width=8)