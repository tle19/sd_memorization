import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModel, AutoImageProcessor

class Embed:
    def __init__(self, cuda):
        self.device = cuda if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
    
    def image_feature(self, input):
        image = Image.open(input)
        image = self.process_image(image)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_feature = self.compute_image_feature(inputs)
        return image_feature
    
class CLIPEmbed(Embed):
    def __init__(self, cuda):
        super().__init__(cuda)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def process_image(self, image):
        image = image.convert("RGB").resize((224, 224))
        return image

    def compute_image_feature(self, inputs):
        image_feature = self.model.get_image_features(**inputs).numpy()
        return image_feature

class DINOEmbed(Embed):
    def __init__(self, cuda):
        super().__init__(cuda)
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    def process_image(self, image):
        image = image.convert("RGB").resize((504, 504))
        return image

    def compute_image_feature(self, inputs):
        image_feature = self.model(**inputs).last_hidden_state.squeeze(0).numpy()
        return image_feature