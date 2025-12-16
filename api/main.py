"""
FastAPI application for deepfake detection service.
"""
import os
import sys
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.predict import DeepfakePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="RESTful API for detecting deepfake images using EfficientNet-B0",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model predictor (loaded on startup)
predictor: Optional[DeepfakePredictor] = None
device: str = "cpu"


# Response models
class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    label: str
    probability: float
    confidence: float
    is_fake: bool
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None


# Startup event
@app.on_event("startup")
async def load_model_on_startup():
    """Load model when API starts."""
    global predictor, device
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.getenv('MODEL_PATH', os.path.join('models', 'best_model.pth'))
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")
        
        predictor = DeepfakePredictor(model_path=model_path, device=device)
        logger.info("✅ Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"❌ Model not found: {str(e)}")
        logger.warning("API will start but prediction endpoints will not work")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.warning("API will start but prediction endpoints will not work")


# Root endpoint
@app.get("/", tags=["General"])
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Deepfake Detection API",
        "status": "active",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "device": device
    }


# Prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict if uploaded image is real or fake.
    
    - **file**: Image file (PNG, JPG, JPEG)
    - Returns prediction with confidence score
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure model file exists."
        )
    
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Supported types: PNG, JPEG, JPG"
        )
    
    try:
        # Read image file
        contents = await file.read()
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: 10MB"
            )
        
        # Open and validate image
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Make prediction
        try:
            result = predictor.predict(image)
            
            # Format response
            response = PredictionResponse(
                label=result['label'],
                probability=result['probability'],
                confidence=round(result['confidence'], 2),
                is_fake=result['is_fake'],
                message=f"Image classified as {result['label'].lower()} with {result['confidence']:.2f}% confidence"
            )
            
            logger.info(f"Prediction: {result['label']} (confidence: {result['confidence']:.2f}%)")
            return response
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during prediction: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Batch prediction endpoint (optional)
@app.post(
    "/predict/batch",
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict multiple images at once.
    
    - **files**: List of image files (PNG, JPG, JPEG)
    - Returns list of predictions
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure model file exists."
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            # Validate file type
            if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                results.append({
                    "filename": file.filename,
                    "error": f"Invalid file type: {file.content_type}"
                })
                continue
            
            # Read image
            contents = await file.read()
            
            # Open image
            try:
                image = Image.open(io.BytesIO(contents)).convert('RGB')
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": f"Invalid image: {str(e)}"
                })
                continue
            
            # Make prediction
            result = predictor.predict(image)
            results.append({
                "filename": file.filename,
                "label": result['label'],
                "probability": result['probability'],
                "confidence": round(result['confidence'], 2),
                "is_fake": result['is_fake']
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Processing error: {str(e)}"
            })
    
    return {
        "total": len(files),
        "processed": len([r for r in results if "error" not in r]),
        "results": results
    }


# Model info endpoint
@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_path": predictor.model_path,
        "device": device,
        "architecture": "EfficientNet-B0",
        "input_size": "224×224×3",
        "status": "loaded"
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )
