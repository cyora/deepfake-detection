"""
Script to run the FastAPI backend.
"""
import uvicorn
import os

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Run uvicorn server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

