import re

phishing_keywords = [
    "verify", "login", "password", "urgent", "update", "security alert",
    "click", "confirm", "bank", "account", "limited time"
]

def extract_email_features(text):
    text = text.lower()
    
    features = {
        "keyword_count": sum(1 for word in phishing_keywords if word in text),
        "num_links": len(re.findall(r'http[s]?://', text)),
        "num_exclamation": text.count("!"),
        "num_digits": sum(c.isdigit() for c in text),
        "contains_html": 1 if "<html>" in text else 0,
    }
    
    return features
