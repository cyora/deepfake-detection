# 🔍 Deepfake Detection System

Deep learning project for detecting deepfake images using the FaceForensics++ dataset.

## 📋 Project Overview

This project implements a binary classification model to detect deepfake images using transfer learning with EfficientNet/ResNet architectures.

### Dataset
- **FaceForensics++**: Contains real and manipulated facial images
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/adham7elmy/faceforencispp-extracted-frames)

## 🏗️ Project Structure
```
deepfake-detection/
├── notebooks/          # Jupyter notebooks (from Kaggle)
├── models/            # Trained model files
├── src/               # Source code
│   ├── model.py       # Model architecture
│   ├── predict.py     # Prediction functions
│   └── utils.py       # Utility functions
├── api/               # FastAPI application
├── frontend/          # Streamlit UI
├── tests/             # Unit and integration tests
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Multi-container setup
└── requirements.txt   # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip
- Docker (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/cyora/deepfake-detection.git
cd deepfake-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Development Status

- [x] Repository setup ✅
- [x] Data exploration (Kaggle) ✅
- [x] Model training (EfficientNet-B0) ✅
- [x] Model saved to Kaggle Dataset ✅
- [ ] Model evaluation on test set (NEXT STEP 🎯)
- [ ] API development
- [ ] Frontend development
- [ ] Testing
- [ ] Docker deployment
- [ ] Documentation


## 🎯 Training Results

### Model Performance
- **Architecture**: EfficientNet-B0 with transfer learning
- **Training Status**: ✅ Complete  
- **Training Duration**: ~25 minutes on Kaggle GPU (Tesla T4)
- **Dataset**: 7,000 images (3,500 real + 3,500 fake)
- **Model Size**: ~21 MB

### Saved Artifacts
- ✅ **Training notebook**: `notebooks/deepfake-detection-training.ipynb`
- ✅ **Trained model**: `models/best_model.pth` (local only)
- ✅ **Training history**: `models/training_history.json`
- ✅ **Kaggle Model Dataset**: [Link](https://www.kaggle.com/datasets/cyrinegraf/deepfake-efficientnet-trained-model)

### Training Configuration
- **Epochs**: 10
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Data Augmentation**: Random flip, rotation, color jitter
- **Train/Val/Test Split**: 70% / 15% / 15%



## 🛠️ Technologies

- **Deep Learning**: PyTorch, torchvision
- **API**: FastAPI
- **Frontend**: Streamlit
- **Experiment Tracking**: MLflow
- **Testing**: pytest
- **Deployment**: Docker

## 📝 Deliverables

- ✅ GitHub repository with clear structure
- ⏳ Trained model (PyTorch/ONNX)
- ⏳ FastAPI service with Swagger docs
- ⏳ User interface (Streamlit)
- ⏳ MLflow experiment tracking
- ⏳ Docker containerization
- ⏳ Unit and integration tests

## 👨‍💻 Author

Cyrine Graf
Arij Karoui 
Ichrak mzita

** Data Science Engineering Students **

## 📄 License

This project is licensed under the MIT License.
