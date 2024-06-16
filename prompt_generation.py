import os
import torch
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None

def blip_model(model_id):
    global model, processor
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    processor = Blip2Processor.from_pretrained(model_id)
    model.to(device)

def generate_prompts(prompts, path, output_path):

    features = [
        'racial background',
        'age',
        'hair color',
        'eye color'
        ]
    
    prefixes = [
        'they are ',
        'they have ',
        'they\'re '
        ]
    
    blip_prompts = ['Question: What is their approximate '] * len(features)

    for i in range(len(features)):
        blip_prompt = blip_prompts[i]
        feature = features[i]
        blip_prompt = blip_prompt + feature + '? Answer:'
        blip_prompts[i] = blip_prompt


    start_val = 0
    counter = '{:0{width}d}'.format(0, width=8)

    generated_prompts = []

    for prompt in prompts:
        print('PROMPT', counter, '-', prompt)

        image_path = os.path.join(path, prompt + '.png')
        image = Image.open(image_path)

        start_val += 1
        counter = '{:0{width}d}'.format(start_val, width=8)

        prompt = "this is a picture of"
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True, max_new_tokens = 150)[0].strip()

        answers = []
        for prompt in blip_prompts:
            inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
            answer_id = model.generate(**inputs)

            generated_answer = processor.batch_decode(answer_id, skip_special_tokens=True, max_new_tokens=150)[0].strip()
            generated_answer = generated_answer.lower().replace('.', '')

            bad_answers = ['I don\'t know', 'unknown']
            if generated_answer == 'I don\'t know':
                generated_answer = 'white'
            answers.append(generated_answer)

        for i in range(len(answers)):
            generated_answer = answers[i]

            #temporary brute force fix for feature prompt syntax
            if i > 1:
                generated_answer = generated_answer + ' ' + features[i]

            if any(pre in generated_answer for pre in prefixes):
                continue
            generated_answer = prefixes[0] + generated_answer

            answers[i] = generated_answer
        
        text = text + ', ' + ', '.join(answers)
        generated_prompts.append(text)
        print(text)

    csv_path = os.path.join(output_path, 'prompts.csv')
    prompts_df = pd.read_csv(csv_path)
    prompts_df['Description'] = generated_prompts
    prompts_df.to_csv(csv_path)

    return generated_prompts

    