import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model
from data_loader import get_data_loaders
from config import LEARNING_RATE, EPOCHS, MODEL_DIR
import os

def train_model(train_dir, val_dir):
    train_loader, val_loader = get_data_loaders(train_dir, val_dir)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_best.pt"))
    print("✅ Model saved!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m src.train <train_dir> <val_dir>")
        sys.exit(1)

    train_dir = sys.argv[1]
    val_dir = sys.argv[2]
    train_model(train_dir, val_dir)
