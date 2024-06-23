import os
import torch
import pandas as pd
from PIL import Image
from transformers import set_seed
from transformers import AutoModelForCausalLM, LlamaTokenizer
from utils import print_title

# chat example
class CaptionGeneration2():

    def __init__(self, model_id, seed, cuda):
        set_seed(seed) 
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained('THUDM/cogvlm-chat-hf', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        self.model.to(self.device).eval()

        self.human_nouns = [
            'man', 'men', 'woman', 'women', 'boy', 'girl', 'he', 'she', 'his', 'her', 'lady',
            'child', 'children', 'adult', 'adults', 'baby', 'babies',
            'person', 'people', 'actor', 'actress', 'lady', 'players'
        ]

    def generate_captions(self, prompts, path, output_path, temp, k, p):
        generated_captions = []
        is_human = []

        for index, prompt in enumerate(prompts):
            image_path = os.path.join(path, prompt + '.png')
            image = Image.open(image_path)

            pre_prompt = 'Describe this image'
            text = self.generate_one_caption(image, pre_prompt, temp=temp, top_k=k, top_p=p, min=30, max=40)

            if any(word in self.human_nouns for word in text.split()):
                is_human.append(True)
            else:
                is_human.append(False)

            generated_captions.append(text)

            print_title('PROMPT', prompt, index)
            print(text)

        csv_path = os.path.join(output_path, 'prompts.csv')
        prompts_df = pd.read_csv(csv_path)
        prompts_df['Description'] = generated_captions
        prompts_df['is_human'] = is_human
        prompts_df.to_csv(csv_path, index=False)

        return generated_captions
    
    def generate_one_caption(self, image, prompt, temp=1.0, top_k=50, top_p=1.0, min=0, max=20):
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]

        return self.tokenizer.decode(outputs[0])
