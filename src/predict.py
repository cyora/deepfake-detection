"""
Prediction functions for deepfake detection.
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import os

from src.model import DeepfakeDetector, load_model


# Image preprocessing pipeline
def get_transform():
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed tensor ready for model
    """
    transform = get_transform()
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def predict_image(model: DeepfakeDetector, 
                 image: Image.Image, 
                 device: str = 'cpu') -> Tuple[float, str, float]:
    """
    Predict if image is real or fake.
    
    Args:
        model: Loaded model
        image: PIL Image object
        device: Device to run inference on
    
    Returns:
        Tuple of (probability, label, confidence)
        - probability: Probability of being fake (0-1)
        - label: "Real" or "Fake"
        - confidence: Confidence score (0-100)
    """
    model.eval()
    
    # Preprocess image
    image_tensor = preprocess_image(image).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        # Apply sigmoid to get probability
        probability = torch.sigmoid(output).item()
    
    # Determine label
    is_fake = probability > 0.5
    label = "Fake" if is_fake else "Real"
    
    # Calculate confidence (distance from 0.5)
    confidence = abs(probability - 0.5) * 2 * 100
    
    return probability, label, confidence


def predict_from_path(model: DeepfakeDetector, 
                     image_path: str, 
                     device: str = 'cpu') -> Tuple[float, str, float]:
    """
    Predict from image file path.
    
    Args:
        model: Loaded model
        image_path: Path to image file
        device: Device to run inference on
    
    Returns:
        Tuple of (probability, label, confidence)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    return predict_image(model, image, device)


class DeepfakePredictor:
    """
    Wrapper class for easy model loading and prediction.
    """
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to model checkpoint. If None, looks for default path.
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        
        if model_path is None:
            # Default model path
            model_path = os.path.join('models', 'best_model.pth')
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join('models', 'best_weights.pth'),
                    os.path.join('models', 'final_weights.pth'),
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please download the model from Kaggle dataset or train it first."
            )
        
        self.model = load_model(model_path, device)
        self.model_path = model_path
    
    def predict(self, image) -> dict:
        """
        Predict if image is real or fake.
        
        Args:
            image: PIL Image object or path to image file
        
        Returns:
            Dictionary with prediction results
        """
        if isinstance(image, str):
            probability, label, confidence = predict_from_path(
                self.model, image, self.device
            )
        else:
            probability, label, confidence = predict_image(
                self.model, image, self.device
            )
        
        return {
            'probability': probability,
            'label': label,
            'confidence': confidence,
            'is_fake': label == "Fake"
        }
