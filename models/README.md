# Trained Models

This folder contains training artifacts and metadata for the deepfake detection model.

## 📁 Files in This Folder

### Files in Git (Small)
- `training_history.json` - Training metrics (loss, accuracy per epoch)
- `curves.png` - Training/validation curves visualization
- `.gitkeep` - Keeps folder structure in Git

### Files NOT in Git (Large - Stored Separately)
These files are **too large** for GitHub (20+ MB each):
- `best_model.pth` - Complete checkpoint (model + optimizer state)
- `best_weights.pth` - Model weights only
- `checkpoint_epoch_X.pth` - Checkpoints from each epoch
- `final_weights.pth` - Final model weights

## 🌐 Where to Find Model Files

**Option 1: Kaggle Dataset (Recommended)**
- Dataset: https://www.kaggle.com/datasets/cyrinegraf/deepfake-efficientnet-trained-model
- Contains: `best_model.pth`, `best_weights.pth`, `training_history.json`
- Free, permanent storage
- Easy to load in Kaggle notebooks

**Option 2: Local Storage**
- If you trained the model, files are saved locally
- Located in this `models/` folder on your machine
- Not synced to GitHub (ignored by `.gitignore`)

## 🚀 How to Use the Model

### Load in Kaggle Notebook
```python
import torch
import torch.nn as nn
import timm

# 1. Add dataset to your notebook:
#    https://www.kaggle.com/datasets/cyrinegraf/deepfake-efficientnet-trained-model

# 2. Define model architecture
class DeepfakeDetector(nn.Module):
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

# 3. Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeDetector()
checkpoint = torch.load('/kaggle/input/deepfake-efficientnet-trained-model/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("✅ Model loaded and ready!")
```

### Load Locally
```python
import torch

# Load model
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 📊 Model Specifications

- **Architecture**: EfficientNet-B0 with custom classifier head
- **Input Size**: 224×224×3 RGB images
- **Output**: Single value (sigmoid → probability)
  - < 0.5 = Real
  - > 0.5 = Fake
- **Parameters**: ~5.3M
- **Model Size**: ~21 MB
- **Framework**: PyTorch 2.1+
- **Dependencies**: `torch`, `timm`, `torchvision`

## 📈 Training Details

- **Dataset**: FaceForensics++ (7,000 images)
- **Split**: 70% train / 15% val / 15% test
- **Epochs**: 10
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss**: BCEWithLogitsLoss
- **Data Augmentation**: Random flip, rotation, color jitter
- **Training Time**: ~25 minutes on Tesla T4 GPU

## 🔒 Why Files Aren't in Git

GitHub has a file size limit of 100 MB, and our model files are:
- `best_model.pth`: ~25 MB
- `best_weights.pth`: ~21 MB
- All checkpoints: ~200+ MB total

Instead, we:
1. Store them locally
2. Upload to Kaggle Dataset (free, permanent)
3. Add `.gitignore` rules to prevent accidental commits

## 📝 Notes

- Always use `best_model.pth` for deployment (best validation loss)
- `training_history.json` contains all metrics for analysis
- See main README for project overview