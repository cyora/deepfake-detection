"""
Streamlit frontend for deepfake detection.
"""
import streamlit as st
import torch
from PIL import Image
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.predict import DeepfakePredictor


# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .fake-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-bar {
        height: 30px;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model with caching."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = DeepfakePredictor(device=device)
        return predictor, device
    except FileNotFoundError as e:
        st.error(f"Model not found: {str(e)}")
        st.info("""
        **To use this application:**
        1. Download the trained model from the [Kaggle Dataset](https://www.kaggle.com/datasets/cyrinegraf/deepfake-efficientnet-trained-model)
        2. Place `best_model.pth` in the `models/` directory
        3. Refresh this page
        """)
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">🔍 Deepfake Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to detect if it\'s real or AI-generated</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a deep learning model (EfficientNet-B0) 
        to detect deepfake images.
        
        **Model Performance:**
        - Accuracy: 99.9%
        - Precision: 100%
        - Recall: 99.8%
        - F1-Score: 99.9%
        
        **How it works:**
        1. Upload an image containing a face
        2. The model analyzes the image
        3. Get instant results with confidence score
        """)
        
        st.header("📊 Model Info")
        st.info("""
        **Architecture:** EfficientNet-B0
        **Input Size:** 224×224×3
        **Dataset:** FaceForensics++
        **Training:** 7,000 images
        """)
        
        st.header("⚠️ Limitations")
        st.warning("""
        - Works best with frontal face images
        - May have false positives/negatives
        - Not 100% accurate
        - For research/educational purposes
        """)
    
    # Load model
    predictor, device = load_model()
    
    if predictor is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing a face to analyze"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            # Use width-based sizing for compatibility across Streamlit versions
            st.image(image, caption="Uploaded Image")
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]}×{image.size[1]} pixels")
    
    with col2:
        st.header("🔬 Analysis Results")
        
        if uploaded_file is not None:
            # Predict button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Make prediction
                        result = predictor.predict(image)
                        
                        # Display results
                        probability = result['probability']
                        label = result['label']
                        confidence = result['confidence']
                        is_fake = result['is_fake']
                        
                        # Result box
                        box_class = "fake-box" if is_fake else "real-box"
                        icon = "⚠️" if is_fake else "✅"
                        color = "#dc3545" if is_fake else "#28a745"
                        
                        st.markdown(f"""
                        <div class="result-box {box_class}">
                            <h2 style="text-align: center; color: {color};">
                                {icon} {label}
                            </h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence score
                        st.subheader("Confidence Score")
                        st.progress(confidence / 100)
                        st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Probability breakdown
                        st.subheader("Probability Breakdown")
                        col_fake, col_real = st.columns(2)
                        
                        with col_fake:
                            fake_prob = probability * 100
                            st.metric("Fake Probability", f"{fake_prob:.2f}%")
                        
                        with col_real:
                            real_prob = (1 - probability) * 100
                            st.metric("Real Probability", f"{real_prob:.2f}%")
                        
                        # Additional info
                        st.info(f"**Device:** {device.upper()}")
                        
                        # Interpretation
                        if confidence > 80:
                            st.success("High confidence prediction")
                        elif confidence > 50:
                            st.warning("Moderate confidence prediction")
                        else:
                            st.error("Low confidence prediction - results may be unreliable")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.exception(e)
        else:
            st.info("👆 Please upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built using Streamlit and PyTorch</p>
        <p>Model trained on FaceForensics++ dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
