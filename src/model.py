"""
Model architecture for deepfake detection.
"""
import torch
import torch.nn as nn
import timm


class DeepfakeDetector(nn.Module):
    """
    Deepfake detection model using EfficientNet-B0 as backbone.
    """
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, 
                                         num_classes=0, global_pool='')
        n_features = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.flatten(1)
        return self.classifier(features)


def load_model(model_path: str, device: str = 'cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in evaluation mode
    """
    model = DeepfakeDetector()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

