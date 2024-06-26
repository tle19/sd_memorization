import os
import torch
import pandas as pd
import spacy
import re
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from utils import print_title, punc_splice

class CaptionGeneration:
    
    def __init__(self, model_id, cuda):
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        self.processor = Blip2Processor.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.nlp = spacy.load('en_core_web_sm')

        self.blip_questions = {
            'Question: What color is their hair? Answer:': "black",
            'Question: What color is their eyes? Answer:': "brown",
            'Question: What is their ethnicity? Answer:': "white",
            'Question: What is their approximate age? Answer:': "35"
        }

        self.human_nouns = [
            'man', 'men', 'woman', 'women', 'boy', 'girl', 'gentleman', 'lady', 
            'child', 'children', 'adult', 'adults', 'baby', 'babies',
            'person', 'people', 'actor', 'actress', 'lady', 'players',
            'singer', 'singers'
        ]

        self.bad_answers = [
            'i don\'t know', 'i do not know', 'i dont know', 'i am not sure', 'i\'m not sure', 
            'unknown', 'mystery', 'it depends', 'it ain\'t', 'i have no idea'
        ]

        self.age_patterns = [
            'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
            'twenties', 'thirties', 'forties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'teen'
        ]
        
    def generate_captions(self, prompts, path, output_path, temp, k, p, beams):
        generated_captions = []
        is_human = []

        for index, prompt in enumerate(prompts):
            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            pre_prompt = "this is a picture of"
            text = self.generate_one_caption(image, pre_prompt, temp, k, p, beams, 30, 40)

            if any(word in self.human_nouns for word in text.split()):
                is_human.append(True)
            else:
                is_human.append(False)
            
            answers = []
            for question in self.blip_questions:
                answer = self.generate_one_caption(image, question, temp, k, p, beams, max=25).lower()
                answer = self.filter_vague(answer, question)

                if 'age' in question:
                    answer = self.extract_age(answer)
                else:
                    answer = self.extract_adjective(answer)
                    
                if not answer:
                    answer = self.blip_questions[question]

                answers.append(answer)

            hair_and_eyes = f'with {answers[0]} hair and {answers[1]} eyes)'
            age_and_ethnicity = f'({answers[3]} year old {answers[2]}'
            
            text = self.add_attribute(text, hair_and_eyes, True)
            text = self.add_attribute(text, age_and_ethnicity)
            re.sub(r'\s+', ' ', text)

            generated_captions.append(text)

            print_title('PROMPT', prompt, index)
            print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_captions
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path, index=False)

        return generated_captions
    
    def generate_one_caption(self, image, prompt, temp, top_k, top_p, num_beams, min=0, max=20):
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, temperature=temp, top_k=top_k, top_p=top_p, num_beams=num_beams, min_length=min, max_length=max, do_sample=True)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = text.lower().replace('.', ',')

        return text
    
    def filter_vague(self, answer, question):
        
        for bad_ans in self.bad_answers:
            if bad_ans in answer:
                default_answer = self.blip_questions[question]
                answer = answer.replace(bad_ans, default_answer)
                break

        return answer
  
    def extract_adjective(self, text):
        proc_text = self.nlp(text)

        for token in proc_text:
            if token.pos_ == 'ADJ' or token.pos_ == 'PROPN':
                return punc_splice(',', token.text)

    def extract_age(self, text):

        proc_text = self.nlp(text)
        
        reg_pattern = r"\b\d+s\b"

        for token in proc_text:
            if token.like_num:
                return token.text
            else:
                if re.search(reg_pattern, token.text):
                    return token.text.replace('s', '')
                for pat in self.age_patterns:
                    if pat in token.text:
                        return token.text
    
    def add_attribute(self, text, adjective, add_modifier=False):
        proc_text = self.nlp(text)

        modified_text = []
        inserted = False
        
        for token in proc_text:
            if token.pos_ == 'NOUN' and not inserted:
                if add_modifier:
                    modified_text.append(token.text)
                    modified_text.append(adjective)
                else:
                    modified_text.append(adjective)
                    modified_text.append(token.text)
                inserted = True
            else:
                modified_text.append(token.text)

        modified_text = ' '.join(modified_text)

        return modified_text