import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, UploadFile, File
from PIL import Image

app = FastAPI()

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
idx_to_class = None

# SAME TRANSFORM USED DURING TRAINING
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.on_event("startup")
def load_model():
    global model, idx_to_class

    print("Loading model...")

    # -------- LOAD CLASS MAPPING --------
    class_json_path = os.path.join(BASE_DIR, "models", "classes.json")
    if not os.path.exists(class_json_path):
        raise FileNotFoundError(f"{class_json_path} not found!")

    with open(class_json_path) as f:
        class_to_idx = json.load(f)

    # reverse mapping (0->cat, 1->dog)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Class mapping:", idx_to_class)

    # -------- LOAD MODEL --------
    model_path = os.path.join(BASE_DIR, "models", "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found!")

    # Create ResNet18 architecture
    model = models.resnet18(weights=None)

    # Replace last layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = idx_to_class[predicted.item()]

    return {
        "prediction": label,
        "confidence": float(confidence.item())
    }
