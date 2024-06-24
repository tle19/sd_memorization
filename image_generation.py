import os
import torch
from transformers import set_seed
from diffusers import StableDiffusionPipeline
from utils import print_title

class ImageGeneration:

    def __init__(self, model_id, seed, cuda):
        set_seed(seed)
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)   
        self.pipe = self.pipe.to(self.device)

    def generate_images(self, names, prompts, sd_folder_path1, num_steps):
        for index, prompt in enumerate(prompts):
            print_title('IMAGE', names[index], index)

            with torch.no_grad():
                image = self.pipe(prompt, num_inference_steps=num_steps, width=1024, height=1024).images[0]

            image_path = os.path.join(sd_folder_path1, names[index] + '.png')
            image.save(image_path)