import os
import torch
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class caption_generation():
    
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        self.processor = Blip2Processor.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model.to(self.device)

        self.nouns = [
            'person', 'people', 'man', 'men' 'woman', 'women',
            'boy', 'girl', 'player', 'players'
            ]
        
        self.subjects = [
            'they are', 'they have', 'they\'re'
            ]

        self.bad_answers = [
            'I don\'t know', 'unknown'
            ]
        
        self.blip_questions = [
            'Question: What is their ethnicity? Answer:',
            'Question: What is their approximate age? Answer:',
            'Question: What is their hair color? Answer:',
            'Question: What is their eye color? Answer:'
            ]
    
    def generate_captions(self, prompts, path, output_path):
        start_val = 0
        counter = '{:0{width}d}'.format(start_val, width=8)

        generated_captions = []
        is_human = []

        for prompt in prompts:
            print('PROMPT', counter, '-', prompt)

            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            start_val += 1
            counter = '{:0{width}d}'.format(start_val, width=8)

            prompt = "this is a picture of"
            text = self.generate_one_caption(image, prompt, 30, 40)
            text = text.replace('.', ',')

            if any(human in text for human in self.nouns):
                is_human.append(True)
            else:
                is_human.append(False)
            
            answers = self.add_questions(image)
            text = text + ', ' + ','.join(answers)

            generated_captions.append(text)

            print(answers)
            # print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_captions
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path, index=False)

        return generated_captions
    
    def generate_one_caption(self, image, prompt, min=0, max=20):
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, min_length=min, max_length=max)
            #experiment with temperature, top_k, top_p

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text.lower()
    
    def add_questions(self, image):
        answers = []

        for question in self.blip_questions:
            answer = self.generate_one_caption(image, question, max=30)
            answer = answer.replace(',', '')

            # if answer in bad_answers:
            #     answer = 'white'
            
            if not any(subject in answer for subject in self.subjects):
                answer = self.subjects[0] + ' ' + answer
            else:
                for subject in self.subjects:
                    if subject in answer:
                        answer = answer.replace(subject, self.subjects[0])
                        break
            
            answers.append(answer)

        return answers