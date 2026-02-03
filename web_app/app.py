import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppress ABSL warnings

from flask import Flask, render_template, request
import tensorflow as tf
import tldextract
import numpy as np
import re

# Load model
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to load combined model first, fallback to URL-only model
try:
    model = tf.keras.models.load_model("../phishing_model_combined.h5")
    scaler_mean = np.load("../scaler_mean_combined.npy")
    scaler_scale = np.load("../scaler_scale_combined.npy")
    model_type = "combined"
    print("✓ Loaded combined URL+Email model")
except FileNotFoundError:
    model = tf.keras.models.load_model("../phishing_model.h5")
    scaler_mean = np.load("../scaler_mean.npy")
    scaler_scale = np.load("../scaler_scale.npy")
    model_type = "url_only"
    print("✓ Loaded URL-only model")

model.compile(optimizer='adam', loss='binary_crossentropy')

app = Flask(__name__)

# Feature extraction functions
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


@app.route("/", methods=["GET", "POST"])
def index():
    result_class = None
    result_message = None
    source_info = None

    if request.method == "POST":
        url = request.form.get("url", "").strip()
        email_text = request.form.get("email_text", "").strip()

        if not url and not email_text:
            result_class = "error"
            result_message = "❌ Please provide at least a URL or email text"
            return render_template("index.html", result_class=result_class, result_message=result_message, source_info=source_info, model_type=model_type)

        # Extract features based on model type
        if model_type == "combined":
            # Combined model expects 14 features (9 URL + 5 Email)
            
            if url and email_text:
                # Both provided
                url_feats = extract_url_features(url)
                email_feats = extract_email_features(email_text)
                features = np.concatenate([url_feats, email_feats])
                source_info = "Analyzing URL + Email"
                
            elif url:
                # URL only - pad email features with zeros
                url_feats = extract_url_features(url)
                email_feats = np.zeros(5)
                features = np.concatenate([url_feats, email_feats])
                source_info = "Analyzing URL only"
                
            else:
                # Email only - extract URL from email or use dummy
                urls_in_email = re.findall(r'http[s]?://[^\s<>"]+', email_text)
                if urls_in_email:
                    url_feats = extract_url_features(urls_in_email[0])
                    source_info = f"Analyzing Email (found URL: {urls_in_email[0][:50]}...)"
                else:
                    url_feats = np.zeros(9)
                    source_info = "Analyzing Email only"
                
                email_feats = extract_email_features(email_text)
                features = np.concatenate([url_feats, email_feats])

        else:
            # URL-only model expects 9 features
            if url:
                features = extract_url_features(url)
                source_info = "Analyzing URL"
            else:
                # Try to extract URL from email
                urls_in_email = re.findall(r'http[s]?://[^\s<>"]+', email_text)
                if urls_in_email:
                    features = extract_url_features(urls_in_email[0])
                    source_info = f"Extracted URL from email: {urls_in_email[0]}"
                else:
                    result_class = "error"
                    result_message = "❌ URL-only model requires a URL"
                    return render_template("index.html", result_class=result_class, result_message=result_message, source_info=source_info, model_type=model_type)

        # Scale and predict
        features_scaled = scale_features(features).reshape(1, -1)
        pred = model.predict(features_scaled, verbose=0)[0][0]
        
        if pred >= 0.5:
            result_class = "malicious"
            result_message = f"⚠️ PHISHING/MALICIOUS - {pred*100:.1f}% confidence"
        else:
            result_class = "benign"
            result_message = f"✅ LEGITIMATE - {(1-pred)*100:.1f}% confidence"

    return render_template("index.html", result_class=result_class, result_message=result_message, source_info=source_info, model_type=model_type)


if __name__ == "__main__":
    app.run(debug=True)
