from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# Define device (Use CPU if no GPU is available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model
model_path = "model/resnet_model.pth"  # Make sure this is in the project folder

# Load the model architecture and weights
model = models.resnet18(pretrained=False)
num_classes = 5  # Number of waste classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  # Modify if needed

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
