import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU (save memory)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from flask import Flask, render_template, request
import tensorflow as tf
import tldextract
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Get absolute paths for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, "phishing_model_combined.h5")
MEAN_PATH = os.path.join(PARENT_DIR, "scaler_mean_combined.npy")
SCALE_PATH = os.path.join(PARENT_DIR, "scaler_scale_combined.npy")

# Lazy load model on first request (saves memory)
model = None
scaler_mean = None
scaler_scale = None
model_type = None

def load_model_once():
    """Lazy load model on first request"""
    global model, scaler_mean, scaler_scale, model_type
    
    if model is not None:
        return
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        scaler_mean = np.load(MEAN_PATH)
        scaler_scale = np.load(SCALE_PATH)
        model_type = "combined"
        print("✓ Loaded combined model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model_type = "error"
        raise

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
    try:
        load_model_once()  # Lazy load on first request
    except Exception as e:
        return render_template("index.html", 
                             result_class="error", 
                             result_message=f"❌ Model loading failed: {str(e)}", 
                             source_info=None, 
                             model_type="error")
    
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


@app.route("/health")
def health():
    """Health check endpoint for deployment platforms"""
    return {"status": "ok", "model_loaded": model is not None}, 200


if __name__ == "__main__":
    app.run(debug=True)
