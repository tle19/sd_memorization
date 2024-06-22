import os
import torch
import spacy
import pandas as pd
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from utils import print_title

class CaptionGeneration():
    
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
        self.processor = Blip2Processor.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.nlp = spacy.load('en_core_web_sm')

        self.blip_questions = {
            'Question: What color is their hair? Answer:': "black",
            'Question: What color is their eyes? Answer:': "brown",
            'Question: What is their ethnicity? Answer:': "white",
            'Question: What is their approximate age? Answer:': "40"
        }

        self.human_nouns = [
            'man', 'men', 'woman', 'women', 'boy', 'girl', 'he', 'she', 'his', 'her',
            'person', 'people', 'actor', 'actress', 'player', 'players', 
            'they', 'them', 'their', 'it'
        ]
        
    def generate_captions(self, prompts, path, output_path):
        generated_captions = []
        is_human = []

        for index, prompt in enumerate(prompts):
            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            pre_prompt = "this is a picture of"
            # pre_prompt = "Question: Can you describe this person? Answer:"
            text = self.generate_one_caption(image, pre_prompt, temp=0.9, min=30, max=40)

            if any(human in text for human in self.human_nouns):
                is_human.append(True)
            else:
                is_human.append(False)
            
            answers = []
            for question in self.blip_questions:
                answer = self.generate_one_caption(image, question, max=15).lower()

                answer = self.filter_vague(answer, question)

                if 'age' in question:
                    answer = self.extract_age(answer)
                else:
                    answer = self.extract_adjective(answer)

                if not answer:
                    answer = self.blip_questions[question]

                answers.append(answer)

            hair_and_eyes = f'with {answers[0]} hair and {answers[1]} eyes'
            ethnicity = answers[2]
            age = f'{answers[3]} year old'
            
            text = self.add_attribute(text, hair_and_eyes, True)
            text = self.add_attribute(text, ethnicity)
            text = self.add_attribute(text, age)

            generated_captions.append(text)

            print_title('PROMPT', prompt, index)
            print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_captions
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path, index=False)

        return generated_captions
    
    def generate_one_caption(self, image, prompt, temp=1.0, k=50, min=0, max=20):
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, temperature=temp, top_k=k, min_length=min, max_length=max, do_sample=True)
            #experiment with temperature, top_k, top_p

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = text.lower().replace('.', ',')
        return text
    
    def filter_vague(self, answer, question):

        bad_answers = [
            'i don\'t know', 'i do not know', 'i dont know', 'i am not sure', 'i\'m not sure', 
            'unknown', 'mystery', 'it depends', 'it ain\'t', 'i have no idea'
        ]
        
        for bad_ans in bad_answers:
            if bad_ans in answer:
                default_answer = self.blip_questions[question]
                answer = answer.replace(bad_ans, default_answer)
                break
        return answer
    
    def extract_adjective(self, text):
        proc_text = self.nlp(text)

        for token in proc_text:
            if token.pos_ == 'ADJ' or token.pos_ == 'PROPN':
                return self.comma_splice(self, token.text)

    def extract_age(self, text):
        proc_text = self.nlp(text)
        age_number = None
        
        for token in proc_text:
            if token.like_num or self.is_age_text(token.text):
                age_number = token.text
                break
        
        return age_number
    
    def is_age_text(self, token):
        age_patterns = [
            'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'teen'
        ]

        if token.text.endswith('s') and token.text[:-1].isdigit():
            return True
        
        for pat in age_patterns:
            if pat in token.text:
                return True
        
        return False
    
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
    
    def comma_splice(self, text):
        pos = text.find(',')
        if pos != -1:
            return text[:pos]
        else:
            return text
        