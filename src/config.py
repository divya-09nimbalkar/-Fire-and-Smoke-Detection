import os

# Paths
DATA_DIR = os.path.join(os.getcwd(), "data")
MODEL_DIR = os.path.join(os.getcwd(), "models", "final")
LOG_DIR = os.path.join(os.getcwd(), "logs")

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Classes
NUM_CLASSES = 3
CLASS_NAMES = ["fire", "smoke", "normal"]
