from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import timm
from PIL import Image
import io
import numpy as np
from torchvision import transforms as T
import os

app = Flask(__name__)
CORS(app)

# Define classes for skin type detection
classes = {
    "dry": 0,
    "normal": 1,
    "oily": 2
}

# Setup model parameters
model_name = "rexnet_150"
im_size = 64
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define your transformations
transform = T.Compose([
    T.Resize((im_size, im_size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

# Load the trained model
model = timm.create_model(model_name=model_name, pretrained=False, num_classes=len(classes))
model_path = os.path.join(os.path.dirname(__file__), "saved_models/skin_best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
            
            class_names = list(classes.keys())
            prediction = class_names[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            
            return jsonify({
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'all_probabilities': {
                    class_name: round(prob * 100, 2)
                    for class_name, prob in zip(
                        class_names, 
                        torch.nn.functional.softmax(outputs, dim=1)[0].tolist()
                    )
                }
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': model_name}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)