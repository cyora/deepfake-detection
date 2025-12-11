"""
FastAPI application for deepfake detection service.
"""
from fastapi import FastAPI

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images",
    version="0.1.0"
)

@app.get("/")
def read_root():
    return {"message": "Deepfake Detection API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# TODO: Implement prediction endpoint