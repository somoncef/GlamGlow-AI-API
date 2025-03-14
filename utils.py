import torch
from torchvision import transforms as T
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import base64

class ImageProcessor:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], im_size=64):
        self.mean = mean
        self.std = std
        self.im_size = im_size
        self.transform = T.Compose([
            T.Resize((im_size, im_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
    def preprocess_image(self, image_bytes):
        """Convert image bytes to tensor for model input"""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(image).unsqueeze(0)
    
    def tensor_to_image(self, tensor):
        """Convert a pytorch tensor to a PIL Image"""
        # Denormalize
        invTrans = T.Compose([
            T.Normalize(mean=[0., 0., 0.], std=[1/s for s in self.std]),
            T.Normalize(mean=[-m for m in self.mean], std=[1., 1., 1.])
        ])
        tensor = invTrans(tensor.squeeze()).cpu()
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

    def get_base64_image(self, tensor):
        """Convert a tensor to a base64 encoded image string"""
        img = self.tensor_to_image(tensor)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

class ResultsVisualizer:
    """Class for visualizing model predictions"""
    
    @staticmethod
    def generate_probability_chart(probabilities, class_names):
        """Generate a bar chart showing prediction probabilities"""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, probabilities, color='skyblue')
        
        # Highlight the highest probability
        highest_idx = np.argmax(probabilities)
        bars[highest_idx].set_color('green')
        
        plt.ylabel('Probability')
        plt.title('Skin Type Prediction Probabilities')
        plt.ylim(0, 1.0)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
