"""
Train combined URL + Email phishing detection model
"""
import pandas as pd
import numpy as np
import re
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ===========================
# 1. LOAD DATASETS
# ===========================
print("Loading datasets...")

# URL dataset
df_url = pd.read_csv("data/clean_dataset.csv")
df_url["label"] = df_url["label"].map({"benign": 0, "malicious": 1})

# Email dataset (after download and cleaning)
try:
    # Try cleaned dataset first
    df_email = pd.read_csv("data/email_dataset_clean.csv")
    df_email["label"] = df_email["label"].map({"legitimate": 0, "phishing": 1})
    print(f"✓ Loaded {len(df_email)} emails from cleaned dataset")
except FileNotFoundError:
    try:
        # Fallback to original
        df_email = pd.read_csv("data/email_dataset.csv")
        df_email.columns = ['email_text', 'label'] if len(df_email.columns) == 2 else df_email.columns
        df_email["label"] = df_email["label"].str.lower().map({
            "legitimate": 0, "phishing": 1, 
            "safe email": 0, "phishing email": 1,
            "ham": 0, "spam": 1
        })
        print(f"✓ Loaded {len(df_email)} emails")
    except FileNotFoundError:
        print("⚠️ Email dataset not found. Run clean_email_dataset.py first")
        print("Training URL-only model...")
        df_email = None

# ===========================
# 2. FEATURE EXTRACTION
# ===========================

def extract_url_features(url):
    """Extract 9 features from URL"""
    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix
    
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        len(domain),
        len(suffix),
        int("https" in url.lower()),
        int(url.startswith("http://")),
        int(url.startswith("https://")),
    ]

def extract_email_features(text):
    """Extract 5 features from email text"""
    text_lower = text.lower()
    
    phishing_keywords = [
        "verify", "login", "password", "urgent", "update", 
        "security alert", "click", "confirm", "bank", 
        "account", "limited time"
    ]
    
    return [
        sum(1 for word in phishing_keywords if word in text_lower),  # keyword count
        len(re.findall(r'http[s]?://', text)),                      # num links
        text.count("!"),                                             # exclamation marks
        sum(c.isdigit() for c in text),                              # digit count
        int("<html>" in text_lower or "<!doctype" in text_lower),   # has HTML
    ]

# ===========================
# 3. BUILD FEATURE MATRIX
# ===========================

if df_email is not None:
    print("\nBuilding combined feature matrix...")
    
    # URL features
    X_url = np.array([extract_url_features(url) for url in df_url["url"]])
    y_url = df_url["label"].values
    
    # Email features (extract URL from email if present, otherwise use dummy)
    X_email_features = []
    y_email = []
    
    for idx, row in df_email.iterrows():
        email_text = str(row['email_text'])
        
        # Extract URL from email (if present)
        urls_in_email = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
        
        if urls_in_email:
            url = urls_in_email[0]  # Use first URL
        else:
            url = "http://example.com"  # Dummy URL for emails without links
        
        # Combine URL + Email features
        url_feats = extract_url_features(url)
        email_feats = extract_email_features(email_text)
        combined = url_feats + email_feats
        
        X_email_features.append(combined)
        y_email.append(row['label'])
    
    X_email = np.array(X_email_features)
    y_email = np.array(y_email)
    
    # Pad URL features with zeros for email features (9 URL + 5 email = 14 total)
    X_url_padded = np.hstack([X_url, np.zeros((X_url.shape[0], 5))])
    
    # Combine both datasets
    X = np.vstack([X_url_padded, X_email])
    y = np.hstack([y_url, y_email])
    
    print(f"Total samples: {len(X)} (URL: {len(X_url)}, Email: {len(X_email)})")
    print(f"Features: {X.shape[1]} (9 URL + 5 Email)")
    
else:
    # URL-only model
    print("\nBuilding URL-only feature matrix...")
    X = np.array([extract_url_features(url) for url in df_url["url"]])
    y = df_url["label"].values
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]} (URL only)")

# ===========================
# 4. PREPROCESSING
# ===========================
print("\nPreprocessing...")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Malicious: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
print(f"Benign: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

# ===========================
# 5. BUILD NEURAL NETWORK
# ===========================
print("\nBuilding model...")

model = Sequential([
    Dense(64, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())

# ===========================
# 6. TRAIN MODEL
# ===========================
print("\nTraining model...")

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ===========================
# 7. EVALUATE MODEL
# ===========================
print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ===========================
# 8. SAVE MODEL
# ===========================
print("\nSaving model...")

if df_email is not None:
    model.save("phishing_model_combined.h5")
    np.save("scaler_mean_combined.npy", scaler.mean_)
    np.save("scaler_scale_combined.npy", scaler.scale_)
    print("✓ Saved combined model (URL + Email)")
else:
    model.save("phishing_model.h5")
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    print("✓ Saved URL-only model")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
