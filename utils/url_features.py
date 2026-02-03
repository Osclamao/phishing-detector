import re
import tldextract

def extract_url_features(url):
    ext = tldextract.extract(url)
    
    domain = ext.domain
    suffix = ext.suffix
    subdomain = ext.subdomain
    
    features = {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "has_https": 1 if url.startswith("https") else 0,
        "has_ip": 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}", url) else 0,
        "suspicious_tld": 1 if suffix in ["xyz", "top", "click", "zip"] else 0,
        "num_subdomains": 0 if subdomain == "" else len(subdomain.split(".")),
        "has_at_symbol": 1 if "@" in url else 0,
    }
    
    return features