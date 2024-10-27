# app/model_loader.py
from ultralytics import YOLO

def load_model(weights_path):
    """Load the YOLO model from the specified weights file."""
    model = YOLO(weights_path, task="detect")
    return model
