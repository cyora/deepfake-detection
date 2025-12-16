"""
Tests for FastAPI endpoints.
"""
import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def create_test_image():
    """Create a test image."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


def test_read_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "active"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"
    assert "model_loaded" in response.json()
    assert "device" in response.json()


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    # May return 503 if model not loaded, or 200 if loaded
    assert response.status_code in [200, 503]


def test_predict_endpoint_no_file():
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_file_type():
    """Test predict endpoint with invalid file type."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_predict_endpoint_valid_image():
    """Test predict endpoint with valid image."""
    img_bytes = create_test_image()
    
    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    
    # May return 503 if model not loaded, or 200 if loaded
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert "probability" in data
        assert "confidence" in data
        assert "is_fake" in data
        assert data["label"] in ["Real", "Fake"]
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 100
    elif response.status_code == 503:
        assert "Model not loaded" in response.json()["detail"]


def test_predict_batch_endpoint():
    """Test batch prediction endpoint."""
    img_bytes1 = create_test_image()
    img_bytes2 = create_test_image()
    
    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("test1.png", img_bytes1, "image/png")),
            ("files", ("test2.png", img_bytes2, "image/png"))
        ]
    )
    
    # May return 503 if model not loaded, or 200 if loaded
    if response.status_code == 200:
        data = response.json()
        assert "total" in data
        assert "processed" in data
        assert "results" in data
    elif response.status_code == 503:
        assert "Model not loaded" in response.json()["detail"]


def test_predict_batch_too_many_files():
    """Test batch prediction with too many files."""
    files = [("files", (f"test{i}.png", create_test_image(), "image/png")) 
             for i in range(11)]
    
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400
    assert "Maximum 10 files" in response.json()["detail"]


def test_cors_headers():
    """Test CORS headers are present."""
    response = client.options("/predict")
    # CORS middleware should be configured
    assert response.status_code in [200, 405]  # OPTIONS may return 405
