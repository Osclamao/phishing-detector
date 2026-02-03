# Email Phishing Datasets Guide

## Available Options

### 1. **SpamAssassin Public Corpus** ⭐ Recommended
**Why:** Free, direct download, well-labeled, ~6,000 emails

**Download:**
```bash
python download_email_dataset.py
```

**Sources:**
- Spam emails: https://spamassassin.apache.org/old/publiccorpus/
- Legitimate emails (ham): Included in corpus
- Format: Raw email text with headers

---

### 2. **Kaggle Email Datasets**
**Why:** Large, pre-cleaned, multiple options

**Popular Datasets:**

#### A. Email Spam Classification Dataset
- **URL:** https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
- **Size:** 5,572 emails
- **Format:** CSV with text and labels
- **Download:**
  ```bash
  pip install kaggle
  python download_kaggle_email_dataset.py
  ```

#### B. Phishing Email Dataset
- **URL:** https://www.kaggle.com/datasets/subhajournal/phishingemails
- **Size:** ~18,000 emails
- **Content:** Real phishing emails collected

#### C. SMS Spam Collection
- **URL:** https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Size:** 5,574 messages
- **Note:** SMS but applicable to emails

---

### 3. **Enron Email Dataset**
**Why:** Largest public email corpus, real business emails

**Details:**
- **Size:** 500,000+ emails
- **Source:** Enron Corporation (released during investigation)
- **URL:** https://www.cs.cmu.edu/~enron/
- **Label:** All legitimate (combine with spam dataset)

**Download:**
```bash
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
```

---

### 4. **CSDMC2010 Spam Corpus**
**Why:** Academic-quality, Chinese and English

**Details:**
- **URL:** http://www.csmining.org/index.php/spam-email-datasets-.html
- **Size:** 4,327 spam + legitimate emails
- **Format:** Raw email files

---

### 5. **Nazario Phishing Corpus**
**Why:** Specific to phishing (not general spam)

**Details:**
- **URL:** https://monkey.org/~jose/phishing/
- **Content:** Real-world phishing emails
- **Update:** Regularly updated with new samples

---

### 6. **GitHub Datasets**

#### Phishing Email Repository
```bash
git clone https://github.com/syedsaqlainhussain/Phishing-Emails-Dataset.git
```

#### Email Security Dataset
```bash
git clone https://github.com/soumit-s/email-spam-dataset.git
```

---

## Quick Start: Recommended Approach

### Option 1: SpamAssassin (Easiest)
```bash
# No API keys needed
python download_email_dataset.py

# Output: data/email_dataset.csv with labeled emails
```

### Option 2: Kaggle (Largest)
```bash
# 1. Setup Kaggle API
# Visit: https://www.kaggle.com/settings/account
# Download: kaggle.json

# 2. Install and download
pip install kaggle
python download_kaggle_email_dataset.py
```

### Option 3: Manual Download
1. Visit Kaggle dataset page
2. Click "Download" button
3. Extract to `data/` folder
4. Rename to `email_dataset.csv`

---

## Dataset Format Requirements

Your CSV should have these columns:
```csv
email_text,label
"Subject: Verify your account...",phishing
"Subject: Meeting tomorrow...",legitimate
```

Or with URL extracted:
```csv
email_text,url,label
"Click here: http://evil.com","http://evil.com",phishing
"Meeting at 3pm","",legitimate
```

---

## Integration Steps

After downloading dataset:

1. **Verify dataset:**
   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/email_dataset.csv'); print(df.head())"
   ```

2. **Update training script** (covered separately)

3. **Extract features** from emails

4. **Retrain model** with combined URL + email features

---

## Dataset Comparison

| Dataset | Size | Format | Labels | Best For |
|---------|------|--------|--------|----------|
| SpamAssassin | ~6K | Raw text | Binary | Quick start |
| Kaggle Spam | ~5K | CSV | Binary | Easy integration |
| Enron | 500K+ | Raw text | Legitimate only | Large legitimate corpus |
| Nazario | ~3K | Raw HTML | Phishing only | Phishing-specific |
| CSDMC2010 | ~4K | Raw text | Binary | Academic research |

---

## Legal & Ethical Notes

✅ **Allowed:**
- Using public datasets for research/education
- Training models for security purposes
- Sharing anonymized results

❌ **Not Allowed:**
- Redistributing original datasets without permission
- Using real user emails without consent
- Creating phishing attacks with trained models

---

## Troubleshooting

### "Dataset too large for memory"
Use chunking:
```python
df = pd.read_csv('data/email_dataset.csv', chunksize=1000)
```

### "Character encoding errors"
Try different encodings:
```python
df = pd.read_csv('data/email_dataset.csv', encoding='latin-1')
# or
df = pd.read_csv('data/email_dataset.csv', encoding='utf-8', errors='ignore')
```

### "Imbalanced dataset"
Use SMOTE or undersampling:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## Next Steps

Once you have a dataset:
1. Run data exploration: `python analyze_email_dataset.py`
2. Update feature extraction: Add email content features
3. Retrain model: `python train_email_model.py`
4. Test predictions: `python predict_email.py`

---

*Last Updated: January 31, 2026*
