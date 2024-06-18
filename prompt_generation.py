import os
import torch
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class prompt_generation():
    
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        self.processor = Blip2Processor.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model.to(self.device)

    def additional_prompts(self):
        
        self.pronoun_list = ['man', 
                        'woman', 
                        'boy', 
                        'girl']

        self.features = [
            'racial background',
            'age',
            'hair color',
            'eye color'
            ]
        
        self.prefixes = [
            'they are ',
            'they have ',
            'they\'re '
            ]
        
        self.blip_prompts = ['Question: What is their approximate '] * len(self.features)

        for i in range(len(self.features)):
            blip_prompt = self.blip_prompts[i]
            feature = self.features[i]
            blip_prompt = blip_prompt + feature + '? Answer:'
            self.blip_prompts[i] = blip_prompt

    def generate_prompts(self, prompts, path, output_path):
        
        self.additional_prompts()

        start_val = 0
        counter = '{:0{width}d}'.format(start_val, width=8)

        generated_prompts = []
        is_human = []

        for prompt in prompts:
            print('PROMPT', counter, '-', prompt)

            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            start_val += 1
            counter = '{:0{width}d}'.format(start_val, width=8)

            prompt = "this is a picture of"
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, min_length=30, max_length=40)
            #experiment with temperature, top_k, top_p

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if any(human in text for human in self.pronoun_list):
                is_human.append(True)
            else:
                is_human.append(False)
            
            # answers = []
            # for prompt in self.blip_prompts:
            #     inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            #     answer_id = self.model.generate(**inputs)

            #     generated_answer = self.processor.batch_decode(answer_id, skip_special_tokens=True, max_new_tokens=150)[0].strip()
            #     generated_answer = generated_answer.lower().replace('.', '')

            #     bad_answers = ['I don\'t know', 'unknown']
            #     if generated_answer in bad_answers:
            #         generated_answer = 'white'

            #     if not any(pre in generated_answer for pre in self.prefixes):
            #         generated_answer = self.prefixes[0] + generated_answer
            #     answers.append(generated_answer)
            
            # text = text + ', ' + ', '.join(answers)
            generated_prompts.append(text)
            print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_prompts
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path)

        return generated_prompts

        