"""
Script to run the Streamlit frontend.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the frontend app path
    frontend_path = os.path.join("frontend", "app.py")
    
    # Run streamlit
    subprocess.run([
        sys.executable, 
        "-m", 
        "streamlit", 
        "run", 
        frontend_path,
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

