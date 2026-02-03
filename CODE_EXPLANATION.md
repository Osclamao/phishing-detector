# Phishing Detection Project - Complete Code Explanation

**Project Overview:** A machine learning-based phishing URL detection system using a shallow neural network. The system classifies URLs as benign or malicious through a web interface or command-line tool.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Data Processing](#data-processing)
4. [Model Training](#model-training)
5. [Prediction Scripts](#prediction-scripts)
6. [Utility Functions](#utility-functions)
7. [Web Application](#web-application)
8. [Usage Workflow](#usage-workflow)

---

## Project Structure

```
phishing_detection_project/
├── clean_data.py              # Data cleaning and merging
├── train_model.py             # Neural network training
├── predict_url.py             # CLI prediction tool
├── main.py                    # Feature extraction tester
├── requirements.txt           # Python dependencies
├── phishing_model.h5          # Trained model (binary)
├── scaler_mean.npy           # Feature scaling parameters
├── scaler_scale.npy          # Feature scaling parameters
├── data/
│   └── clean_dataset.csv     # Cleaned training data
├── model/
│   └── shallow_nn.py         # Reusable model builder
├── utils/
│   ├── url_features.py       # URL feature extraction
│   ├── email_features.py     # Email feature extraction
│   └── preprocess.py         # Feature combination
└── web_app/
    ├── app.py                # Flask web server
    ├── templates/
    │   └── index.html        # Web interface HTML
    └── static/
        └── style.css         # Web interface styling
```

---

## Dependencies

**File:** `requirements.txt`

```
numpy          - Numerical computing and array operations
pandas         - Data manipulation and CSV handling
scikit-learn   - Machine learning utilities (StandardScaler, train_test_split)
tensorflow     - Deep learning framework for neural networks
tldextract     - Parses URLs to extract domain components
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Data Processing

### `clean_data.py` - Dataset Cleaning and Merging

**Purpose:** Combines multiple phishing datasets and standardizes them for training.

**Key Operations:**

1. **Load Two Datasets:**
   - `urldata.csv` - Contains benign/malicious labels
   - `malicious_phish.csv` - Contains phishing/malware/defacement labels

2. **Standardize Columns:**
   - Renames label column to consistent "label"
   - Keeps only `url` and `label` columns

3. **Binary Classification:**
   - Converts all labels to either "benign" or "malicious"
   - Groups phishing, malware, defacement → "malicious"

4. **Data Cleaning:**
   - Merges both datasets using `pd.concat()`
   - Removes duplicate URLs using `drop_duplicates()`

5. **Save Output:**
   - Exports to `clean_dataset.csv`

**Code Flow:**
```python
# Load → Standardize → Simplify Labels → Merge → Remove Duplicates → Save
df1 = pd.read_csv("data/urldata.csv")
df2 = pd.read_csv("data/malicious_phish.csv")
↓
Rename columns to "url" and "label"
↓
Convert all non-benign to "malicious"
↓
df = pd.concat([df1, df2], ignore_index=True)
↓
df = df.drop_duplicates(subset="url")
↓
Save to clean_dataset.csv
```

---

## Model Training

### `train_model.py` - Neural Network Training Pipeline

**Purpose:** Trains a shallow neural network to classify URLs as benign or malicious.

#### Step 1: Load and Prepare Data
```python
df = pd.read_csv("data/clean_dataset.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})
```
- Loads cleaned dataset
- Converts text labels to binary: benign=0, malicious=1

#### Step 2: Feature Extraction
**Function:** `extract_features(url)`

Extracts 9 numerical features from each URL:

| Feature | Description | Example |
|---------|-------------|---------|
| `len(url)` | Total URL length | 45 |
| `url.count('.')` | Number of dots | 3 |
| `url.count('-')` | Number of hyphens | 2 |
| `url.count('@')` | Number of @ symbols | 0 |
| `len(domain)` | Domain name length | 10 |
| `len(suffix)` | TLD length (com, net, etc.) | 3 |
| `"https" in url` | HTTPS presence (boolean) | True→1 |
| `url.startswith("http://")` | HTTP protocol check | False→0 |
| `url.startswith("https://")` | HTTPS protocol check | True→1 |

**Uses `tldextract` library:**
```python
extracted = tldextract.extract("https://example.com")
# extracted.domain = "example"
# extracted.suffix = "com"
```

#### Step 3: Feature Scaling
```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```
- Normalizes features to have mean=0, std=1
- Prevents features with larger ranges from dominating
- Saves `scaler.mean_` and `scaler.scale_` for later use

#### Step 4: Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
- 80% training data, 20% testing data
- `random_state=42` ensures reproducibility

#### Step 5: Neural Network Architecture
```python
model = Sequential([
    Dense(32, activation="relu", input_shape=(9,)),  # Input layer
    Dense(16, activation="relu"),                    # Hidden layer
    Dense(1, activation="sigmoid")                   # Output layer
])
```

**Architecture Details:**
- **Layer 1:** 32 neurons, ReLU activation
  - ReLU: f(x) = max(0, x) - removes negative values
  - Learns complex patterns from 9 input features
  
- **Layer 2:** 16 neurons, ReLU activation
  - Learns higher-level abstractions
  
- **Layer 3:** 1 neuron, Sigmoid activation
  - Sigmoid: f(x) = 1/(1+e^(-x)) - outputs probability [0,1]
  - Output ≥0.5 → malicious, <0.5 → benign

**Loss Function:** Binary crossentropy
- Measures error between predicted probability and actual label
- Formula: -[y*log(ŷ) + (1-y)*log(1-ŷ)]

**Optimizer:** Adam
- Adaptive learning rate optimization
- Automatically adjusts learning speed

#### Step 6: Training
```python
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```
- **Epochs:** 10 passes through entire dataset
- **Batch size:** 32 samples processed together
- **Validation split:** 20% of training data for validation

#### Step 7: Save Model
```python
model.save("phishing_model.h5")              # Neural network
np.save("scaler_mean.npy", scaler.mean_)     # Scaling mean
np.save("scaler_scale.npy", scaler.scale_)   # Scaling std dev
```

---

## Prediction Scripts

### `predict_url.py` - Command-Line URL Checker

**Purpose:** Test individual URLs using the trained model.

**Workflow:**
```python
1. Load model and scaler parameters
   ↓
2. User inputs URL
   ↓
3. Extract 9 features (same as training)
   ↓
4. Scale features: (features - mean) / scale
   ↓
5. model.predict() → probability
   ↓
6. Display result:
   - ≥0.5 → "MALICIOUS (X% confidence)"
   - <0.5 → "LEGIT (X% confidence)"
```

**Key Code:**
```python
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load resources
model = load_model("phishing_model.h5")
scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")

# Predict
features_scaled = (features - scaler_mean) / scaler_scale
prediction = model.predict(features_scaled.reshape(1, -1))[0][0]
```

**Example Usage:**
```
$ python predict_url.py
Enter URL to check: https://secure-paypal-verify.xyz/login
⚠️ https://secure-paypal-verify.xyz/login is MALICIOUS (87.32% confidence)
```

---

### `main.py` - Feature Extraction Testing Tool

**Purpose:** Development/debugging tool to test feature extraction without making predictions.

**What it does:**
1. Prompts for URL and email text
2. Calls utility functions to extract features
3. Combines features from both sources
4. Displays extracted feature dictionary

**Use case:** Verify that feature extraction logic works correctly before training.

---

## Utility Functions

### `utils/url_features.py` - URL Feature Extractor

**Function:** `extract_url_features(url)`

**Returns dictionary with 8 features:**

```python
{
    "url_length": 45,           # Total characters
    "num_dots": 3,              # Dot count
    "num_hyphens": 2,           # Hyphen count
    "has_https": 1,             # 1=HTTPS, 0=HTTP
    "has_ip": 0,                # 1=IP address, 0=domain
    "suspicious_tld": 0,        # 1=.xyz/.top/.click/.zip
    "num_subdomains": 2,        # e.g., "www.mail" = 2
    "has_at_symbol": 0          # 1=contains @, 0=no @
}
```

**Special Detection:**
- **IP Address Check:** Uses regex to detect numeric IPs
  ```python
  re.match(r"^\d{1,3}(\.\d{1,3}){3}", url)
  ```
- **Suspicious TLDs:** Flags uncommon/malicious TLDs
- **Subdomain Count:** Splits subdomain string by dots

**Note:** This produces slightly different features than `train_model.py` (dictionary vs array). Currently used only by `main.py` for testing.

---

### `utils/email_features.py` - Email Content Analyzer

**Function:** `extract_email_features(text)`

**Returns dictionary with 5 features:**

```python
{
    "keyword_count": 3,         # Phishing keywords found
    "num_links": 2,             # HTTP/HTTPS links
    "num_exclamation": 5,       # Exclamation marks
    "num_digits": 42,           # Numeric characters
    "contains_html": 1          # HTML detected
}
```

**Phishing Keywords List:**
```python
["verify", "login", "password", "urgent", "update", 
 "security alert", "click", "confirm", "bank", 
 "account", "limited time"]
```

**Detection Logic:**
- **Keywords:** Case-insensitive substring search
- **Links:** Regex `r'http[s]?://'`
- **HTML:** Checks for `<html>` tag
- **Digits:** Counts all numeric characters

**Use Case:** Can be extended to analyze phishing emails (not currently integrated with model).

---

### `utils/preprocess.py` - Feature Combiner

**Function:** `combine_features(url_features, email_features)`

**Simple utility:**
```python
def combine_features(url_features, email_features):
    combined = {}
    combined.update(url_features)  # Merge URL features
    combined.update(email_features) # Merge email features
    return combined
```

**Purpose:** Merges two feature dictionaries into one for multi-source analysis.

---

## Web Application

### `web_app/app.py` - Flask Web Server

**Purpose:** Provides web interface for URL checking.

#### Setup and Configuration
```python
# Suppress TensorFlow/ABSL warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
absl.logging.set_verbosity(absl.logging.ERROR)

# Load model and scaler
model = tf.keras.models.load_model("../phishing_model.h5")
scaler_mean = np.load("../scaler_mean.npy")
scaler_scale = np.load("../scaler_scale.npy")

app = Flask(__name__)
```

#### Feature Extraction
Identical to `train_model.py` - extracts same 9 features to ensure compatibility.

#### Route Handler
```python
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        
        # Extract → Scale → Predict
        features = extract_features(url)
        features_scaled = (features - scaler_mean) / scaler_scale
        pred = model.predict(features_scaled, verbose=0)[0][0]
        
        # Determine result
        if pred >= 0.5:
            result_class = "malicious"
            result_message = f"⚠️ MALICIOUS - {pred*100:.1f}% confidence"
        else:
            result_class = "benign"
            result_message = f"✅ BENIGN - {(1-pred)*100:.1f}% confidence"
    
    return render_template("index.html", 
                         result_class=result_class, 
                         result_message=result_message)
```

**GET Request:** Displays empty form  
**POST Request:** Processes URL and returns result

#### Running the Server
```python
if __name__ == "__main__":
    app.run(debug=True)  # Runs on http://localhost:5000
```

---

### `web_app/templates/index.html` - Web Interface

**Structure:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Detection Web App</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>Phishing URL Detector</h1>
        
        <!-- Input Form -->
        <form method="POST">
            <input type="text" name="url" 
                   placeholder="Enter a URL here..." required>
            <button type="submit">Check URL</button>
        </form>
        
        <!-- Result Display (conditional) -->
        {% if result_message %}
            <div class="result {{ result_class }}">
                {{ result_message }}
            </div>
        {% endif %}
    </div>
</body>
</html>
```

**Jinja2 Templating:**
- `{% if result_message %}` - Shows result only after submission
- `{{ result_class }}` - Injects "benign" or "malicious" as CSS class
- `{{ result_message }}` - Displays confidence message

---

### `web_app/static/style.css` - Interface Styling

**Key Styling Features:**

```css
/* Page Layout */
body {
    background: #f5f5f5;       /* Light gray background */
    font-family: Arial;
}

/* Centered Container */
.container {
    width: 500px;
    margin: 70px auto;          /* Center horizontally */
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0px 0px 10px #ccc;
}

/* Input Field */
input[type="text"] {
    width: 100%;
    padding: 12px;
    font-size: 16px;
}

/* Submit Button */
button {
    width: 100%;
    padding: 12px;
    background: #0066ff;        /* Blue */
    color: white;
    border: none;
    cursor: pointer;
}

button:hover {
    background: #0047b3;        /* Darker blue on hover */
}

/* Result Box */
.result {
    margin-top: 20px;
    padding: 15px;
    font-size: 20px;
    text-align: center;
    color: white;
}

.result.benign {
    background: #28a745;        /* Green for safe */
}

.result.malicious {
    background: #dc3545;        /* Red for danger */
}
```

**Design Principles:**
- Clean, minimal interface
- Color-coded feedback (green=safe, red=danger)
- Responsive button hover states
- Card-style container with shadow

---

## Usage Workflow

### Complete Project Setup

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare Data**
```bash
python clean_data.py
```
- Merges and cleans datasets
- Creates `data/clean_dataset.csv`

**3. Train Model**
```bash
python train_model.py
```
- Trains neural network
- Creates:
  - `phishing_model.h5`
  - `scaler_mean.npy`
  - `scaler_scale.npy`

**4. Test Predictions**

**Option A: Command Line**
```bash
python predict_url.py
Enter URL to check: https://example.com
✅ https://example.com is LEGIT (92.45% confidence)
```

**Option B: Web Interface**
```bash
cd web_app
python app.py
# Visit http://localhost:5000 in browser
```

---

## Technical Details

### Why Shallow Neural Network?

**Advantages:**
- Fast training (seconds, not hours)
- Low computational requirements
- Good for structured/tabular data with few features
- Less prone to overfitting than deep networks

**When Deep Networks Needed:**
- Image/video processing
- Natural language processing
- Complex pattern recognition
- Large datasets with thousands of features

### Feature Engineering Rationale

**URL Length:** Phishing URLs often longer (try to mimic legitimate sites)

**Dot Count:** Excessive subdomains suspicious (e.g., `login.secure.verify.paypal.com.evil.xyz`)

**HTTPS Check:** Absence of HTTPS can indicate lower security

**Domain/TLD Analysis:** Uncommon TLDs (.xyz, .top) more common in phishing

**Special Characters:** @ symbol can hide real domain (`http://legitimate.com@evil.com`)

### Model Performance Tips

**Improving Accuracy:**
1. **More training data** - Collect larger datasets
2. **Additional features** - SSL certificate age, WHOIS info, page content
3. **Ensemble methods** - Combine multiple models
4. **Hyperparameter tuning** - Adjust layers, neurons, epochs
5. **Regular updates** - Retrain with new phishing examples

---

## File Dependencies

```
predict_url.py          →  phishing_model.h5, scaler_*.npy
web_app/app.py          →  phishing_model.h5, scaler_*.npy
train_model.py          →  data/clean_dataset.csv
clean_data.py           →  data/urldata.csv, data/malicious_phish.csv
main.py                 →  utils/url_features.py, utils/email_features.py
```

---

## Potential Improvements

### 1. Multi-Source Integration
Currently, email features extracted but not used in model. Could train combined URL+email classifier.

### 2. Real-Time Threat Intelligence
Integrate with:
- Google Safe Browsing API
- VirusTotal API
- Certificate transparency logs

### 3. Advanced Features
- SSL certificate validation
- Domain age (WHOIS lookup)
- Page content analysis
- Redirect chain detection
- JavaScript analysis

### 4. Model Enhancements
- Use Random Forest or XGBoost
- Implement model versioning
- A/B testing framework
- Confidence threshold tuning

### 5. Production Deployment
- Containerize with Docker
- Deploy to cloud (AWS, Azure, GCP)
- Add authentication
- Implement rate limiting
- Database for logging predictions

---

## Troubleshooting

### Common Issues

**1. TensorFlow Warnings**
```python
# Already handled in code with:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

**2. Model Loading Errors**
- Ensure `phishing_model.h5` exists
- Check TensorFlow version compatibility
- Verify working directory

**3. Feature Count Mismatch**
- Model expects exactly 9 features
- Ensure `extract_features()` consistent across training/prediction

**4. Web App Can't Find Model**
```python
# app.py uses relative paths:
model = load_model("../phishing_model.h5")
# Must run from web_app/ directory
```

---

## Summary

This phishing detection system demonstrates a complete machine learning pipeline:

1. **Data Pipeline:** Collection → Cleaning → Storage
2. **Model Pipeline:** Feature Engineering → Training → Validation
3. **Deployment:** CLI tool + Web interface
4. **Maintenance:** Retraining capability, modular code

**Key Strengths:**
- Simple, interpretable features
- Fast predictions (<100ms)
- Easy to deploy and maintain
- Extensible architecture

**Learning Outcomes:**
- Binary classification with neural networks
- Feature extraction from URLs
- Web application development with Flask
- Model serialization and deployment
- Full ML project lifecycle

---

*Generated: January 30, 2026*
*Project: Phishing URL Detection System*
