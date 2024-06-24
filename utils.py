import torch
from PIL import Image
from transformers import set_seed
from transformers import CLIPModel, CLIPProcessor
from transformers import ViTImageProcessor, ViTModel

class CLIPEmbed:

    def __init__(self, seed, cuda):
        set_seed(seed)
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def image_feature(self, input):
        image = Image.open(input)
        image = image.convert("RGB").resize((224, 224))
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_feature = self.model.get_image_features(**inputs).numpy()

        return image_feature 

class DINOEmbed:

    def __init__(self, seed, cuda):
        set_seed(seed)
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.model = ViTModel.from_pretrained("facebook/dino-vitb16")
        self.processor = ViTImageProcessor.from_pretrained("facebook/dino-vitb16")
    
    def image_feature(self, input):
        image = Image.open(input)
        image = image.convert("RGB").resize((384, 384))
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_feature = self.model(**inputs).last_hidden_state.squeeze(0).numpy()
        
        return image_feature


def print_title(typ, name, index):
    counter = '{:0{width}d}'.format(index, width=8)
    print('\033[1m' + typ, counter, '-', name + '\033[0m')

def punc_splice(punc, text):
    pos = text.find(punc)
    if pos != -1:
        return text[:pos]
    else:
        return text
    
def generate_graph():
    pass