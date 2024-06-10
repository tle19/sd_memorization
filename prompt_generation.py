from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None

def blip_model(model_id):
    global processor, model
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)


def generate_prompts(prompts, path, output_path):

    start_val = 0
    counter = '{:0{width}d}'.format(0, width=8)

    generated_prompts = []

    for prompt in prompts:
        print('PROMPT', counter, '-', prompt)

        image_path = os.path.join(path, prompt + '.png')
        image = Image.open(image_path)

        start_val += 1
        counter = '{:0{width}d}'.format(start_val, width=8)

        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True, max_new_tokens = 150)[0].strip()
        
        generated_prompts.append(text)
        print(text)

    csv_path = os.path.join(output_path, 'prompts.csv')
    prompts_df = pd.read_csv(csv_path)
    prompts_df['Description'] = generated_prompts
    prompts_df.to_csv(csv_path)

    return generated_prompts

    