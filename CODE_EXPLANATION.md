# Phishing Detection Project - Complete Code Explanation **Project Overview:** A machine learning-based phishing URL detection system using a shallow neural network. The system classifies URLs as benign or malicious through a web interface or command-line tool. --- ## Table of Contents 1.

pasted

the project will take too much time to zip. the project is phishing detection; CHAPTER THREE
MATERIALS AND METHODS
3.1 Methodology
3.2 System Analysis
3.2 1 Analysis of the existing system (Including the architecture of the existing system)
3.2.2 Disadvantages of the existing system
3.3 System Design
3.3.1 The Proposed System (including the architecture of the proposed system)
3.3.2 Algorithm/Flowchart/use case diagram or any design tool for the design of the proposed system
3.3.3 Advantages of the proposed System
3.4  Database Specification

Show more
May 19
Here's a well-structured Chapter Three based on your phishing detection project:

CHAPTER THREE: MATERIALS AND METHODS
3.1 Methodology
This project adopted a quantitative, experimental methodology based on supervised machine learning. The development followed a structured pipeline approach encompassing data collection, preprocessing, feature engineering, model training, evaluation, and deployment. The methodology was guided by the Knowledge Discovery in Databases (KDD) process, which provides a systematic framework for extracting useful patterns from data.

The system was developed using Python as the primary programming language, leveraging libraries such as TensorFlow, Scikit-learn, Pandas, NumPy, and TLDExtract. A shallow neural network was selected as the classification model due to its suitability for structured, tabular data with a relatively small feature set. The final system was deployed both as a command-line interface (CLI) tool and as a Flask-based web application.

3.2 System Analysis
3.2.1 Analysis of the Existing System
Several phishing detection approaches currently exist in practice. These include:

i. Blacklist-Based Systems Traditional anti-phishing tools such as Google Safe Browsing and browser-integrated blockers rely on maintained blacklists of known malicious URLs. When a user visits a URL, it is checked against a pre-compiled database of flagged addresses.

Architecture of the Existing System (Blacklist-Based):

User enters URL
       ↓
Browser/Extension sends URL to blacklist server
       ↓
Server checks URL against known malicious URL database
       ↓
Match found? → Block/Warn user
No match?   → Allow access
ii. Rule-Based/Heuristic Systems Some systems apply manually crafted rules (e.g., flagging URLs with IP addresses, excessive subdomains, or suspicious TLDs) to determine legitimacy.

iii. Traditional Machine Learning Systems Earlier solutions employed classifiers such as Naïve Bayes, Decision Trees, or Support Vector Machines (SVMs) trained on static datasets with hand-engineered features.

3.2.2 Disadvantages of the Existing System
Limitation	Description
Reactive, not proactive	Blacklists only catch URLs that have already been reported; brand-new phishing URLs evade detection
High maintenance cost	Blacklists must be continuously updated by human analysts
Zero-day blindness	Newly registered phishing domains are invisible until flagged
Limited scalability	Rule-based systems fail to adapt as attackers evolve their tactics
High false negatives	Sophisticated phishing URLs mimicking legitimate sites pass heuristic checks
No confidence scoring	Binary block/allow decisions offer no probabilistic insight
Context ignorance	Existing systems do not analyse URL structure features dynamically
3.3 System Design
3.3.1 The Proposed System
The proposed system is a machine learning-based phishing URL detection system built on a shallow neural network. Unlike blacklist or rule-based systems, it generalises patterns learned from thousands of labelled URLs to classify new, previously unseen URLs in real time.

The system accepts a URL as input, extracts nine (9) structural and lexical features from it, scales those features, passes them through a trained neural network, and returns a classification — benign or malicious — alongside a confidence score.

Architecture of the Proposed System:

         ┌─────────────────────────────────┐
         │        User Interface           │
         │  (Web App / CLI Terminal)       │
         └────────────────┬────────────────┘
                          │ URL Input
                          ▼
         ┌─────────────────────────────────┐
         │      Feature Extraction         │
         │  (url_length, dot_count,        │
         │   has_https, has_ip,            │
         │   suspicious_tld, etc.)         │
         └────────────────┬────────────────┘
                          │ 9 Numerical Features
                          ▼
         ┌─────────────────────────────────┐
         │     Feature Scaling             │
         │  StandardScaler (mean=0, std=1) │
         └────────────────┬────────────────┘
                          │ Normalised Feature Vector
                          ▼
         ┌─────────────────────────────────┐
         │     Shallow Neural Network      │
         │  Input(9) → Dense(32,ReLU)      │
         │  → Dense(16,ReLU)               │
         │  → Dense(1, Sigmoid)            │
         └────────────────┬────────────────┘
                          │ Probability Score [0,1]
                          ▼
         ┌─────────────────────────────────┐
         │       Classification            │
         │  ≥ 0.5 → MALICIOUS (⚠️)         │
         │  < 0.5 → BENIGN    (✅)         │
         └─────────────────────────────────┘
3.3.2 Algorithm / Flowchart / Use Case
System Algorithm
START
  │
  ├─ TRAINING PHASE:
  │   1. Load raw datasets (urldata.csv, malicious_phish.csv)
  │   2. Standardise and merge datasets
  │   3. Remove duplicate URLs
  │   4. Extract 9 features from each URL
  │   5. Apply StandardScaler normalisation
  │   6. Split data: 80% train / 20% test
  │   7. Train shallow neural network (10 epochs, batch=32)
  │   8. Evaluate accuracy on test set
  │   9. Save model (.h5) and scaler parameters (.npy)
  │
  └─ PREDICTION PHASE:
      1. Accept URL input from user
      2. Extract 9 features from URL
      3. Load saved scaler → normalise features
      4. Load saved model → run prediction
      5. IF prediction ≥ 0.5 → classify as MALICIOUS
         ELSE → classify as BENIGN
      6. Display result with confidence percentage
END
System Flowchart
         [Start]
            │
     [Enter URL Input]
            │
  [Extract URL Features]
  (length, dots, hyphens,
   HTTPS, IP, TLD, etc.)
            │
  [Normalise Features via
     StandardScaler]
            │
  [Feed into Neural Network]
            │
   [Output Probability Score]
            │
      Score ≥ 0.5?
       /         \
     YES          NO
      │            │
 [MALICIOUS]   [BENIGN]
 [+ Confidence] [+ Confidence]
      │            │
       \          /
        [Display Result]
            │
          [End]
Use Case Diagram (Textual Representation)
Actor	Use Case
End User	Enter URL for scanning
End User	View prediction result (benign/malicious)
End User	View confidence score
System	Extract URL features
System	Normalise features
System	Run neural network prediction
System	Return classification result
Developer	Train model on labelled dataset
Developer	Update/retrain model
Developer	Deploy web application
3.3.3 Advantages of the Proposed System
Proactive Detection — Classifies URLs based on structural patterns, not prior blacklist entries; capable of detecting zero-day phishing links.
Real-Time Classification — Predictions are returned in under 100 milliseconds, making it suitable for live browsing environments.
Confidence Scoring — Unlike binary block/allow systems, the model outputs a probability score, giving users transparency about the certainty of a result.
Adaptability — The model can be periodically retrained with new data to adapt to evolving phishing techniques.
Dual Interface — Available both as a web application (accessible via browser) and a CLI tool (suitable for developers and automated pipelines).
Lightweight Architecture — The shallow neural network requires minimal computational resources compared to deep learning alternatives, enabling deployment on low-spec infrastructure.
Extensibility — The modular codebase allows easy integration of additional features (e.g., email content analysis, WHOIS data, SSL certificate validation) without restructuring the core system.
3.4 Database Specification
The system relies on flat-file storage (CSV and NumPy binary formats) rather than a relational database, given the nature of the ML pipeline. The key data assets are described below:

Training Dataset — clean_dataset.csv
Field	Data Type	Description	Example
url	String	Full URL string	https://example.com/login
label	String → Integer	Class label (benign=0, malicious=1)	malicious / 0 or 1
Source: Merged from urldata.csv and malicious_phish.csv
Preprocessing: Duplicates removed, labels standardised to binary
Label Distribution: Benign (0) and Malicious (1)
Model Artefacts
File	Format	Description
phishing_model.h5	HDF5 Binary	Trained neural network weights and architecture
scaler_mean.npy	NumPy Binary	Mean values used for feature normalisation
scaler_scale.npy	NumPy Binary	Standard deviation values for normalisation
Feature Vector Specification
The model operates on the following 9 input features per URL:

#	Feature Name	Type	Description
1	url_length	Integer	Total number of characters in the URL
2	dot_count	Integer	Number of dot (.) characters
3	hyphen_count	Integer	Number of hyphen (-) characters
4	at_symbol_count	Integer	Number of @ characters
5	domain_length	Integer	Length of extracted domain name
6	suffix_length	Integer	Length of the TLD (e.g., "com" = 3)
7	has_https	Binary (0/1)	Whether "https" appears in the URL
8	starts_with_http	Binary (0/1)	Whether URL begins with "http://"
9	starts_with_https	Binary (0/1)	Whether URL begins with "https://"
All features are normalised using StandardScaler before being passed to the model:

Formula: X_scaled = (X − μ) / σ

Where μ is the mean and σ is the standard deviation computed from the training set and saved to scaler_mean.npy and scaler_scale.npy respectively.

This chapter has outlined the full materials and methods used in developing the phishing detection system — from data preparation and system analysis through to the neural network architecture, design artefacts, and data specifications. Chapter Four will present the implementation, testing, and results of the system.

it should be in a ms word document; font 12, double line spacing, times new roman, justified

May 19

Ran 2 commands, viewed a file, created a file
Ran 2 commands, viewed a file, created a file
Your Chapter Three document is ready! Here's what's included, formatted in Times New Roman, font 12, double line spacing, justified:

3.1 Methodology — KDD approach, tools and rationale
3.2 System Analysis — Analysis of blacklist, rule-based, and traditional ML systems, with the existing system architecture described in a table, plus a full disadvantages table
3.3 System Design — Proposed system architecture table, algorithm (training + prediction phases), flowchart table, use case table, and advantages
3.4 Database Specification — Dataset fields, model artefacts, and all 9 features in a formatted table
Download and open it directly in Microsoft Word. Let me know if you need any section adjusted!


Chapter three phishing detection
Document · DOCX 
starting from the use case table, give them headings

May 20

Viewed 2 files
Viewed 2 files
I'll add proper table headings (e.g. "Table 1:", "Table 2:") starting from the Use Case table onward.


Edited 4 files, ran a command, read a file
Edited 4 files, ran a command, read a file
Done! The four tables now have proper centred bold headings:

Table 1: Use Case Descriptions of the Proposed System
Table 2: Training Dataset Field Specification (clean_dataset.csv)
Table 3: Model Artefact Files and Their Descriptions
Table 4: Feature Vector Specification for Model Input
Let me know if you'd like any other changes!


Chapter three phishing detection
Document · DOCX 
how do i start it on the terminal

3:28 PM
To run the Flask web app, open your terminal and follow these steps:

1. Navigate to the web_app folder

bash
cd path/to/phishing_detection_project/web_app
2. Install dependencies (if not done yet)

bash
pip install -r ../requirements.txt
3. Run the app

bash
python app.py
4. Open your browser and go to:

http://localhost:5000
To use the CLI prediction tool instead:

bash
cd path/to/phishing_detection_project
python predict_url.py
Then just type any URL when prompted.

Common issues:

If python doesn't work, try python3
Make sure phishing_model.h5 exists before running — if not, train the model first with python train_model.py
Run app.py from inside the web_app/ folder specifically, since it uses relative paths to find the model files






Claude is AI and can make mistakes. Please double-check responses.
Pasted content
18.79 KB •709 lines
Formatting may be inconsistent from source
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
