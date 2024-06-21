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
            'man', 'men', 'woman', 'women', 'boy', 'girl',
            'person', 'people', 'player', 'players'
            ]
        
        self.subjects = {
            'they are': "", 'they have': ["hair", "eyes"], 'they\'re': "", 'it\'s': "", 'i am': ""
            }

        self.bad_answers = [
            'i don\'t know', 'it depends', 'i am not sure', 'i\'m not sure', 'unknown', 'mystery'
            ]
        
        self.blip_questions = {
            'Question: What is their ethnicity? Answer:': "white",
            'Question: What is their approximate age? Answer:': "40",
            'Question: What color is their hair? Answer:': "black",
            'Question: What color is their eyes? Answer:': "brown"
            }
            #   dictionary for default values for question prompts
    
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
            text = self.generate_one_caption(image, prompt, temp=0.7, k=35, min=30, max=40)
            text = text.replace('.', ',')

            if any(human in text for human in self.nouns):
                is_human.append(True)
            else:
                is_human.append(False)
            
            answers = self.add_questions(image)
            text = text + ', ' + ', '.join(answers)

            generated_captions.append(text)

            print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_captions
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path, index=False)

        return generated_captions
    
    def generate_one_caption(self, image, prompt, temp, k=50, min=0, max=20):
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, temperature=temp, top_k=k, min_length=min, max_length=max, do_sample=True)
            #experiment with temperature, top_k, top_p

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text.lower()
    
    def comma_splice(self, text):
        pos = text.find(',')
        if pos != -1:
            return text[:pos]
        else:
            return text
    
    def filter_vague(self, text):
        for bad_ans in self.bad_answers:
            if bad_ans in text:
                default_answer = self.blip_questions[text]
                text = text.replace(bad_ans, default_answer)
        return text

    def add_questions(self, image):
        answers = []

        for question in self.blip_questions:
            answer = self.generate_one_caption(image, question, temp=0.6, max=25).lower()

            self.filter_vague(question)

            if not any(subject in answer for subject in self.subjects):
                answer = list(self.subjects.keys())[0] + ' ' + answer
            else:
                for subject in self.subjects:
                    if subject in answer:
                        answer = answer.replace(subject, list(self.subjects.keys())[0])
                        break
            
            for subject in self.subjects:
                features = self.subjects[subject]
                if len(features):
                    for feature in features:
                        if feature in question:
                            answer = answer.replace(list(self.subjects.keys())[0], subject)
                            answer = f"{answer} {feature}"
                            break

            answer = self.comma_splice(answer)

            answers.append(answer)

        return answers