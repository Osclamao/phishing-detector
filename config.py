"""
Configuration settings for Phishing Detection Web Application
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if it exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'").strip('"')
                        os.environ[key] = val
    except Exception as e:
        print(f"⚠️ Error parsing .env file: {e}")

# API Credentials (read from environment variables)
# Google Safe Browsing API Key
# Get one from: https://console.cloud.google.com/apis/library/safebrowsing.googleapis.com
SAFE_BROWSING_API_KEY = os.environ.get("SAFE_BROWSING_API_KEY", "")

# Caching Configuration
# Supported types: 'file' (simple JSON file caching) or 'redis' (production Redis server)
CACHE_TYPE = os.environ.get("PHISH_CACHE_TYPE", "file")
CACHE_FILE_PATH = os.environ.get("PHISH_CACHE_FILE", str(BASE_DIR / "data" / "cache.json"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_EXPIRY_SECONDS = int(os.environ.get("PHISH_CACHE_EXPIRY", "86400")) # Default 24 hours

# Rate Limiting Settings
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("PHISH_RATE_LIMIT_MAX", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("PHISH_RATE_LIMIT_WINDOW", "3600")) # 1 hour

# SSL Certificate Checking Settings
SSL_TIMEOUT_SECONDS = 5.0

# Safe Browsing Client Name / Version
SAFE_BROWSING_CLIENT_ID = "phishing-detector-webapp"
SAFE_BROWSING_CLIENT_VERSION = "2.0.0"

# Offline Mode / Fallback Config
# True to force offline pattern matching mode (does not hit external APIs)
OFFLINE_MODE = os.environ.get("PHISH_OFFLINE_MODE", "false").lower() == "true"
