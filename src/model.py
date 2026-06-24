import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES   # ✅ fixed import

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
