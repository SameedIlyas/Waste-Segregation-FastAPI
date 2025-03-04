from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
import requests

app = FastAPI()

# Define device (Use CPU if no GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_DIR = "model"
MODEL_FILENAME = "resnet_model.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# GitHub raw file URL (Important: Use the `raw.githubusercontent.com` link)
MODEL_URL = "https://raw.githubusercontent.com/SameedIlyas/Waste-Segregation-FastAPI/main/model/resnet_model.pth"

# Function to download model if not available
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        else:
            raise Exception(f"Failed to download model. HTTP Status Code: {response.status_code}")

# Ensure model is available before loading
download_model()

# Load the model architecture and weights
model = models.resnet18(pretrained=None)  # pretrained=False is deprecated
num_classes = 5  # Number of waste classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Define data transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    return class_names[predicted_class.item()]

# FastAPI endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class = predict_image(image_bytes)
    return {"class": predicted_class}
