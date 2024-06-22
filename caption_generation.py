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

        self.nouns = [
            'man', 'men', 'woman', 'women', 'boy', 'girl', 'he', 'she', 'his', 'her',
            'person', 'people', 'player', 'players', 'they', 'them', 'their', 'it'
            ]
        
        self.subjects = {
            'they are': "", 'they have': ["hair", "eyes"], 
            'they\'re': "", 'it\'s': "", 'i am': ""
            }

        self.bad_answers = [
            'i don\'t know', 'i do not know', 'i dont know', 'i am not sure', 'i\'m not sure', 
            'unknown', 'mystery', 'it depends', 'it ain\'t', 'i have no idea'
            'it is not a question for you to answer', 
            ]
        
        self.blip_questions = {
            'Question: What is their ethnicity? Answer:': "white",
            # 'Question: What is their approximate age? Answer:': "40",
            # 'Question: What color is their hair? Answer:': "black",
            # 'Question: What color is their eyes? Answer:': "brown"
            }
            #   dictionary for default values for question prompts
    
    def generate_captions(self, prompts, path, output_path):
        generated_captions = []
        is_human = []

        for index, prompt in enumerate(prompts):
            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            pre_prompt = "this is a picture of"
            text = self.generate_one_caption(image, pre_prompt, temp=0.9, min=30, max=40)

            if any(human in text for human in self.nouns):
                is_human.append(True)
            else:
                is_human.append(False)
            
            answers = self.add_questions(image)
            text = text + ', ' + ', '.join(answers)

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

    def add_questions(self, image):
        answers = []

        for question in self.blip_questions:
            answer = self.generate_one_caption(image, question, max=25).lower()

            answer = self.filter_vague(answer, question)
            adjectives = self.extract_adjectives(answer)
            if adjectives:
                answer = self.add_adjective(answer, adjectives[0])
            # if not any(subject in answer for subject in self.subjects):
            #     answer = list(self.subjects.keys())[0] + ' ' + answer
            # else:
            #     for subject in self.subjects:
            #         if subject in answer:
            #             answer = answer.replace(subject, list(self.subjects.keys())[0])
            #             break
            
            # for subject in self.subjects:
            #     features = self.subjects[subject]
            #     if len(features):
            #         for feature in features:
            #             if feature in question:
            #                 answer = answer.replace(list(self.subjects.keys())[0], subject)
            #                 answer = f"{answer} {feature}"
            #                 break

            answer = self.comma_splice(answer)

            answers.append(answer)

        return answers
    
    def filter_vague(self, answer, question):
        for bad_ans in self.bad_answers:
            if bad_ans in answer:
                default_answer = self.blip_questions[question]
                answer = answer.replace(bad_ans, default_answer)
                break
        return answer
    
    def extract_adjectives(self, text):
        processed_text = self.nlp(text)

        adjectives = []
        for token in processed_text:
            if token.pos_ == 'ADJ' or token.pos_ == 'PROPN':
                adjectives.append(token.text)
    
        return adjectives
    
    def add_adjective(self, text, adjective):
        processed_text = self.nlp(text)

        modified_text = []
        adjective_inserted = False
        
        for token in processed_text:
            if token.pos_ == 'NOUN' and not adjective_inserted:
                modified_text.append(adjective)
                modified_text.append(token.text)
                adjective_inserted = True
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
        