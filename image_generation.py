import os
import torch
from diffusers import StableDiffusionPipeline
from utils import print_title

class ImageGeneration:

    def __init__(self, model_id, num_steps, cuda):
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)   
        self.pipe = self.pipe.to(self.device)
        self.num_steps = num_steps

    def generate_images(self, names, prompts, image_folder):
        for index, prompt in enumerate(prompts):
            print_title('IMAGE', names[index], index)

            with torch.no_grad():
                image = self.pipe(prompt, num_inference_steps=self.num_steps, width=512, height=512).images[0]

            image_path = os.path.join(image_folder, names[index] + '.png')
            image.save(image_path)