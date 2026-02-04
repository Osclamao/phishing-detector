"""
WSGI entry point for production deployment
"""
import os
# Set environment before importing app
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from app import app

# Don't load model at startup - let it lazy load on first request

if __name__ == "__main__":
    app.run()
