import torch
from torchvision import transforms
from PIL import Image
from model import build_model
from config import MODEL_DIR, CLASS_NAMES, IMG_SIZE
import os

def load_model():
    model = build_model()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_best.pt"), map_location="cpu"))
    model.eval()
    return model

def predict_image(image_path):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]
