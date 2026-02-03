"""
Predict phishing from URL and/or Email using combined model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re
import tldextract
from tensorflow.keras.models import load_model

# ===========================
# LOAD MODEL
# ===========================
try:
    model = load_model("phishing_model_combined.h5")
    scaler_mean = np.load("scaler_mean_combined.npy")
    scaler_scale = np.load("scaler_scale_combined.npy")
    print("✓ Loaded combined URL+Email model")
    model_type = "combined"
except FileNotFoundError:
    model = load_model("phishing_model.h5")
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")
    print("✓ Loaded URL-only model")
    model_type = "url_only"

model.compile(optimizer='adam', loss='binary_crossentropy')

# ===========================
# FEATURE EXTRACTION
# ===========================

def extract_url_features(url):
    """Extract 9 features from URL"""
    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix
    
    return np.array([
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        len(domain),
        len(suffix),
        int("https" in url.lower()),
        int(url.startswith("http://")),
        int(url.startswith("https://")),
    ])

def extract_email_features(text):
    """Extract 5 features from email text"""
    text_lower = text.lower()
    
    phishing_keywords = [
        "verify", "login", "password", "urgent", "update", 
        "security alert", "click", "confirm", "bank", 
        "account", "limited time"
    ]
    
    return np.array([
        sum(1 for word in phishing_keywords if word in text_lower),
        len(re.findall(r'http[s]?://', text)),
        text.count("!"),
        sum(c.isdigit() for c in text),
        int("<html>" in text_lower or "<!doctype" in text_lower),
    ])

def scale_features(features):
    """Scale features using saved scaler"""
    return (features - scaler_mean) / scaler_scale

# ===========================
# PREDICTION
# ===========================

print("\n" + "="*60)
print("PHISHING DETECTION - Combined URL + Email")
print("="*60)

# Get input
url = input("\nEnter URL (or press Enter to skip): ").strip()
email_text = input("Paste email text (or press Enter to skip): ").strip()

if not url and not email_text:
    print("❌ Please provide at least a URL or email text")
    exit()

# Extract features
if model_type == "combined":
    # Combined model expects 14 features (9 URL + 5 Email)
    
    if url and email_text:
        # Both provided
        url_feats = extract_url_features(url)
        email_feats = extract_email_features(email_text)
        features = np.concatenate([url_feats, email_feats])
        source = "URL + Email"
        
    elif url:
        # URL only - pad email features with zeros
        url_feats = extract_url_features(url)
        email_feats = np.zeros(5)
        features = np.concatenate([url_feats, email_feats])
        source = "URL only"
        
    else:
        # Email only - extract URL from email or use dummy
        urls_in_email = re.findall(r'http[s]?://[^\s<>"]+', email_text)
        if urls_in_email:
            url_feats = extract_url_features(urls_in_email[0])
        else:
            url_feats = np.zeros(9)
        
        email_feats = extract_email_features(email_text)
        features = np.concatenate([url_feats, email_feats])
        source = "Email only"

else:
    # URL-only model expects 9 features
    if url:
        features = extract_url_features(url)
        source = "URL"
    else:
        # Try to extract URL from email
        urls_in_email = re.findall(r'http[s]?://[^\s<>"]+', email_text)
        if urls_in_email:
            features = extract_url_features(urls_in_email[0])
            source = f"URL from email: {urls_in_email[0]}"
        else:
            print("❌ URL-only model requires a URL")
            exit()

# Scale and predict
features_scaled = scale_features(features).reshape(1, -1)
prediction = model.predict(features_scaled, verbose=0)[0][0]

# Display result
print("\n" + "="*60)
print(f"Source: {source}")
print("="*60)

if prediction >= 0.5:
    print(f"⚠️  PHISHING/MALICIOUS")
    print(f"Confidence: {prediction*100:.1f}%")
    print("\n⚠️  WARNING: This appears to be a phishing attempt!")
    print("   - Do NOT click any links")
    print("   - Do NOT enter personal information")
    print("   - Report as spam/phishing")
else:
    print(f"✅ LEGITIMATE")
    print(f"Confidence: {(1-prediction)*100:.1f}%")
    print("\n✓ This appears safe, but always verify:")
    print("  - Check sender email address")
    print("  - Verify domain authenticity")
    print("  - Be cautious with sensitive information")

print("="*60)
