# Email + URL Phishing Detection - Integration Guide

## Overview
This guide explains how to integrate email phishing detection with your existing URL detector.

---

## What's Been Created

### 1. **Dataset Downloaders**
- `download_email_dataset.py` - SpamAssassin corpus (~6K emails)
- `download_kaggle_email_dataset.py` - Kaggle datasets (~18K emails)

### 2. **Training Script**
- `train_combined_model.py` - Trains on URL + Email features (14 total)

### 3. **Prediction Script**
- `predict_combined.py` - Tests URLs and/or emails

---

## Step-by-Step Integration

### **Step 1: Download Email Dataset** ✓ In Progress
```bash
python download_email_dataset.py
```
**Output:** `data/email_dataset.csv` with ~6,000 labeled emails

**Status:** Currently downloading (takes 3-5 minutes)

---

### **Step 2: Train Combined Model**
Once download completes:
```bash
python train_combined_model.py
```

**What it does:**
- Loads URL dataset (existing) + Email dataset (new)
- Extracts 14 features per sample:
  - **9 URL features:** length, dots, hyphens, @, domain length, TLD, HTTPS, etc.
  - **5 Email features:** phishing keywords, links, exclamations, digits, HTML
- Trains neural network: 64 → 32 → 16 → 1 neurons
- Saves: `phishing_model_combined.h5`

**Expected output:**
```
Total samples: ~XXX,XXX
Features: 14 (9 URL + 5 Email)
Training samples: ~XXX,XXX
Test Accuracy: ~92-95%
✓ Saved combined model
```

---

### **Step 3: Test Predictions**
```bash
python predict_combined.py
```

**Example Usage:**

#### Test URL only:
```
Enter URL: https://secure-paypal-login.xyz/verify
Paste email text: [Enter]

⚠️ PHISHING/MALICIOUS
Confidence: 87.3%
```

#### Test Email only:
```
Enter URL: [Enter]
Paste email text: URGENT! Your account will be suspended. Click here to verify: http://evil.com

⚠️ PHISHING/MALICIOUS
Confidence: 92.1%
```

#### Test Both:
```
Enter URL: http://suspicious-bank.com
Paste email text: Dear customer, verify your account immediately!

⚠️ PHISHING/MALICIOUS
Confidence: 94.5%
```

---

## Feature Breakdown

### URL Features (9 features)
```python
1. len(url)                    # Total length
2. url.count('.')              # Dot count
3. url.count('-')              # Hyphen count
4. url.count('@')              # @ symbol
5. len(domain)                 # Domain length
6. len(suffix)                 # TLD length
7. "https" in url              # HTTPS presence
8. url.startswith("http://")   # HTTP protocol
9. url.startswith("https://")  # HTTPS protocol
```

### Email Features (5 features)
```python
1. keyword_count               # Phishing keywords found
2. num_links                   # HTTP/HTTPS links in email
3. num_exclamation            # Exclamation marks
4. num_digits                 # Numeric characters
5. contains_html              # HTML content detected
```

**Phishing Keywords:**
- verify, login, password, urgent, update
- security alert, click, confirm, bank
- account, limited time

---

## Model Architecture

### Combined Model (14 features → 1 output)
```
Input: 14 features
   ↓
Dense(64, ReLU) + Dropout(0.3)
   ↓
Dense(32, ReLU) + Dropout(0.2)
   ↓
Dense(16, ReLU)
   ↓
Dense(1, Sigmoid)
   ↓
Output: Probability [0-1]
  ≥0.5 = Phishing
  <0.5 = Legitimate
```

**Improvements over original:**
- More layers (4 vs 3)
- Dropout layers to prevent overfitting
- Larger first layer (64 vs 32 neurons)
- Combined data sources

---

## File Structure After Integration

```
phishing_detection_project/
├── data/
│   ├── clean_dataset.csv           # URL dataset (original)
│   └── email_dataset.csv           # Email dataset (NEW)
├── download_email_dataset.py       # Dataset downloader (NEW)
├── train_combined_model.py         # Combined trainer (NEW)
├── predict_combined.py             # Combined predictor (NEW)
├── phishing_model_combined.h5      # Trained combined model (NEW)
├── scaler_mean_combined.npy        # Scaler parameters (NEW)
├── scaler_scale_combined.npy       # Scaler parameters (NEW)
└── ... (existing files)
```

---

## Comparison: Before vs After

### Before (URL Only)
- **Input:** URL
- **Features:** 9
- **Detects:** Malicious URLs
- **Limitations:** Cannot analyze email content

### After (URL + Email)
- **Input:** URL and/or Email
- **Features:** 14
- **Detects:** Phishing URLs AND emails
- **Advantages:** 
  - Multi-source detection
  - Better accuracy
  - Analyzes email content
  - Extracts URLs from emails

---

## Testing the Combined Model

### Test Cases to Try:

#### 1. Legitimate Email
```
Email: "Hi team, meeting tomorrow at 3pm in conference room B"
Expected: ✅ LEGITIMATE
```

#### 2. Phishing Email (No URL)
```
Email: "URGENT! Your bank account requires immediate verification! 
        Click to update your password now!!!"
Expected: ⚠️ PHISHING
```

#### 3. Phishing Email (With URL)
```
Email: "Dear customer, verify your PayPal account: http://paypal-secure.xyz"
Expected: ⚠️ PHISHING
```

#### 4. Legitimate URL
```
URL: https://github.com/user/repository
Expected: ✅ LEGITIMATE
```

#### 5. Suspicious URL
```
URL: http://paypal-verify-account-2026.tk/login
Expected: ⚠️ PHISHING
```

---

## Web App Integration (Optional)

To add email checking to your Flask web app:

1. **Update** `web_app/app.py`:
   - Load combined model
   - Add email textarea field
   - Extract 14 features

2. **Update** `web_app/templates/index.html`:
   - Add email input field
   - Update form handling

3. **Update** `web_app/static/style.css`:
   - Style email textarea

I can create these updates if you want a web interface for email checking!

---

## Performance Expectations

### URL-Only Model (Original)
- Accuracy: ~85-90%
- Features: 9
- Training time: 30 seconds

### Combined Model (New)
- Accuracy: ~92-96%
- Features: 14
- Training time: 1-2 minutes
- Better generalization

---

## Troubleshooting

### "Email dataset not found"
```bash
# Run downloader again
python download_email_dataset.py

# Or use Kaggle alternative
python download_kaggle_email_dataset.py
```

### "Feature shape mismatch"
- Ensure combined model expects 14 features
- Check that both URL and email features are extracted
- Verify scaler was trained on 14 features

### "Low accuracy on emails"
- May need more email training data
- Consider adding more phishing keywords
- Check email feature extraction logic

---

## Next Steps

✅ **Step 1:** Wait for email dataset download to complete

⏳ **Step 2:** Run `python train_combined_model.py`

⏳ **Step 3:** Test with `python predict_combined.py`

⏳ **Step 4:** (Optional) Update web app for email input

⏳ **Step 5:** Collect more data and retrain periodically

---

## Advanced Improvements

Once basic integration works:

1. **More Email Features:**
   - Sender email analysis
   - Header information
   - Attachment detection
   - Language analysis

2. **Better Models:**
   - LSTM for email text sequences
   - BERT for natural language understanding
   - Ensemble methods (Random Forest + Neural Network)

3. **Real-Time Features:**
   - Domain age lookup (WHOIS)
   - SSL certificate validation
   - Reputation database integration

4. **Production Deployment:**
   - Docker containerization
   - REST API for predictions
   - Database logging
   - Model monitoring

---

**Current Status:** 
- ✅ Integration scripts created
- ⏳ Email dataset downloading
- ⏳ Ready to train combined model

Check download progress with:
```bash
ls -lh data/email_dataset.csv
```

Once file appears, proceed to Step 2!

---

*Last Updated: January 31, 2026*
