import os
import torch
import pandas as pd
import spacy
import re
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from utils import print_title, punc_splice

class CaptionGeneration:
    
    def __init__(self, model_id, temp, top_k, top_p, num_beams, cuda):
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        self.processor = Blip2Processor.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model.to(self.device)

        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
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

        self.age_patterns = [
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'eightteen', 'nineteen',
            'twenty', 'thirty', 'forty', 'fourty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
            'twenties', 'thirties', 'forties', 'fourties', 'fifties', 'sixties', 'seventies', 'eighties', 'nineties'
        ]

        self.bad_answers = [
            'i don\'t know', 'i do not know', 'i dont know', 'i am not sure', 'i\'m not sure', 
            'unknown', 'mystery', 'it depends', 'it ain\'t', 'i have no idea'
        ]
        
    def generate_captions(self, prompts, path, output_path):
        generated_captions = []
        is_human = []

        for index, prompt in enumerate(prompts):
            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            pre_prompt = "this is a picture of"
            text = self.generate_one_caption(
                image, 
                pre_prompt, 
                self.temp, 
                self.top_k, 
                self.top_p, 
                self.num_beams, 
                min_length=30, 
                max_length=40
            )

            if any(word in self.human_nouns for word in text.split()):
                is_human.append(True)
            else:
                is_human.append(False)
            
            if is_human[-1]:
                text = self.additional_attributes(image, text)

            generated_captions.append(text)

            print_title('PROMPT', prompt, index)
            print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_captions
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path, index=False)

        return generated_captions
    
    def generate_one_caption(self, image, prompt, temp, top_k, top_p, num_beams, min_length=0, max_length=20):
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(
            **inputs, 
            temperature=temp, 
            top_k=top_k, 
            top_p=top_p, 
            num_beams=num_beams, 
            min_length=min_length, 
            max_length=max_length, 
            do_sample=True
        )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = text.lower().replace('.', ',')

        return text
    
    def additional_attributes(self, image, text):
        answers = []
        for question in self.blip_questions:
            answer = self.generate_one_caption(
                image, 
                question, 
                self.temp, 
                self.top_k, 
                self.top_p, 
                self.num_beams, 
                max_length=25
            )
            
            answer = self.filter_vague(answer, question)

            if "ethnicity" in question:
                answer = self.extract_ethnicity(answer)
            elif 'age' in question:
                answer = self.extract_age(answer)
            else:
                answer = self.extract_adjective(answer)
                
            if not answer:
                answer = self.blip_questions[question]

            answers.append(answer)

        hair_and_eyes = f'with {answers[0]} hair and {answers[1]} eyes'
        age_and_ethnicity = f'{answers[3]} year old {answers[2]}'
        
        text = self.add_attribute(text, hair_and_eyes, True)
        text = self.add_attribute(text, age_and_ethnicity)

        return text
    
    def filter_vague(self, answer, question):
        for bad_ans in self.bad_answers:
            if bad_ans in answer:
                default_answer = self.blip_questions[question]
                answer = answer.replace(bad_ans, default_answer)
                break

        return answer
  
    def extract_adjective(self, text):
        doc = self.nlp(text)

        for token in doc:
            if token.pos_ == 'ADJ':
                return token.text
                    
    def extract_ethnicity(self, text):
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in {"NORP", "LANGUAGE", "GPE"}:
                return ent.text
            
    def extract_age(self, text):
        digit_pattern = r"(\d+)s?"
        num_pattern = r'(' + '|'.join(self.age_patterns) + r')\s?-?(' + '|'.join(self.age_patterns) + ')?'

        match = re.search(digit_pattern, text)
        if match:
            substring = match.group(1)
            return substring

        match = re.search(num_pattern, text)
        if match:
            substring = match.group()
            return substring
    
    def add_attribute(self, text, attribute, after=False):
        text_split = text.split()

        insert_index = 0

        for i, word in enumerate(text_split):
            if word in self.human_nouns:
                insert_index = i
                if after:
                    insert_index += 1
                break

        text_split.insert(insert_index, attribute)    
        modified_text = ' '.join(text_split)

        return modified_text