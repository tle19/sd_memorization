import os
import torch
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator

class image_generation():

    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device =  Accelerator().device if torch.cuda.device_count() > 1 else self.device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)
        self.pipe = self.pipe.to(self.device)

    def generate_images(self, names, prompts, sd_folder_path1):
        start_val = 0
        counter = '{:0{width}d}'.format(start_val, width=8)

        for prompt in prompts:
            print('IMAGE', counter, '-', names[start_val])
            
            if self.accelerator:
                with self.accelerator.autocast():
                    image = self.pipe(prompt).images[0]
            else:
                with torch.autocast(self.device):
                    image = self.pipe(prompt).images[0]

            image_path = os.path.join(sd_folder_path1, names[start_val] + '.png')
            image.save(image_path)

            start_val += 1
            counter = '{:0{width}d}'.format(start_val, width=8)