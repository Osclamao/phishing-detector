"""
URL Verification Module: Handles Google Safe Browsing, SSL Validation, WHOIS query, and caching.
"""
import os
import re
import socket
import ssl
import json
import time
import urllib.parse
from datetime import datetime
import requests
import config

# Simple Cache Implementation (Supports File-based and Redis fallback)
class DetectionCache:
    def __init__(self):
        self.cache_type = config.CACHE_TYPE
        self.file_path = config.CACHE_FILE_PATH
        self.redis_url = config.REDIS_URL
        self.expiry = config.CACHE_EXPIRY_SECONDS
        self._redis_client = None

        if self.cache_type == "redis":
            try:
                import redis
                self._redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                print("✓ Connected to Redis Cache")
            except Exception as e:
                print(f"⚠️ Redis cache connection failed: {e}. Falling back to file cache.")
                self.cache_type = "file"
        
        if self.cache_type == "file":
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            if not os.path.exists(self.file_path):
                self._write_file_cache({})

    def _read_file_cache(self) -> dict:
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading cache file: {e}")
        return {}

    def _write_file_cache(self, data: dict):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error writing cache file: {e}")

    def get(self, key: str) -> dict:
        if config.OFFLINE_MODE:
            return None
            
        if self.cache_type == "redis" and self._redis_client:
            try:
                cached = self._redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"⚠️ Redis cache get error: {e}")
        else:
            data = self._read_file_cache()
            if key in data:
                entry = data[key]
                if time.time() < entry.get("expires_at", 0):
                    return entry.get("value")
                else:
                    # Clean up expired
                    del data[key]
                    self._write_file_cache(data)
        return None

    def set(self, key: str, value: dict):
        if config.OFFLINE_MODE:
            return
            
        if self.cache_type == "redis" and self._redis_client:
            try:
                self._redis_client.setex(key, self.expiry, json.dumps(value))
            except Exception as e:
                print(f"⚠️ Redis cache set error: {e}")
        else:
            data = self._read_file_cache()
            data[key] = {
                "expires_at": time.time() + self.expiry,
                "value": value
            }
            self._write_file_cache(data)

cache = DetectionCache()


# Google Safe Browsing Integration
def check_google_safe_browsing(url: str, api_key: str = None) -> dict:
    """
    Checks URL against Google Safe Browsing Lookup API (v4)
    """
    if not api_key:
        api_key = config.SAFE_BROWSING_API_KEY
        
    if not api_key:
        return {"status": "unconfigured", "is_flagged": False, "reason": "No API Key configured"}

    api_url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "client": {
            "clientId": config.SAFE_BROWSING_CLIENT_ID,
            "clientVersion": config.SAFE_BROWSING_CLIENT_VERSION
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=6.0)
        response.raise_for_status()
        res_data = response.json()
        
        if "matches" in res_data and len(res_data["matches"]) > 0:
            match = res_data["matches"][0]
            return {
                "status": "success",
                "is_flagged": True,
                "threat_type": match.get("threatType"),
                "platform": match.get("platformType")
            }
        
        return {"status": "success", "is_flagged": False}
    except Exception as e:
        print(f"⚠️ Safe Browsing API error: {e}")
        return {"status": "error", "is_flagged": False, "reason": str(e)}


# SSL Validation
def validate_ssl(hostname: str) -> dict:
    """
    Validates the SSL certificate for a given hostname.
    """
    if not hostname:
        return {"has_ssl": False, "valid": False, "reason": "No hostname"}

    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    
    # Clean port if any
    hostname_clean = hostname.split(':')[0]

    try:
        # Connect to port 443
        with socket.create_connection((hostname_clean, 443), timeout=config.SSL_TIMEOUT_SECONDS) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname_clean) as ssock:
                cert = ssock.getpeercert()
                
                # Expiry date parsing: "notAfter": 'Nov 23 23:59:59 2026 GMT'
                not_after_str = cert.get("notAfter")
                expiry_dt = None
                days_to_expire = None
                
                if not_after_str:
                    try:
                        expiry_dt = datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z")
                        days_to_expire = (expiry_dt - datetime.utcnow()).days
                    except Exception as parse_err:
                        print(f"⚠️ SSL Expiry date parsing error: {parse_err}")
                
                # Get subject and issuer brand
                issuer = dict(x[0] for x in cert.get("issuer", ()))
                issuer_name = issuer.get("organizationName", issuer.get("commonName", "Unknown"))
                
                subject = dict(x[0] for x in cert.get("subject", ()))
                subject_name = subject.get("commonName", "Unknown")

                return {
                    "has_ssl": True,
                    "valid": True,
                    "issuer": issuer_name,
                    "subject": subject_name,
                    "days_to_expire": days_to_expire,
                    "expired": days_to_expire is not None and days_to_expire < 0,
                    "not_after": not_after_str
                }
    except ssl.SSLCertVerificationError as cert_err:
        return {
            "has_ssl": True,
            "valid": False,
            "reason": "SSL Verification Failed: Certificate is self-signed, untrusted, or has hostname mismatch.",
            "error_code": "VERIFICATION_ERROR"
        }
    except ssl.SSLError as ssl_err:
        return {
            "has_ssl": True,
            "valid": False,
            "reason": f"SSL Handshake Error: {ssl_err}",
            "error_code": "HANDSHAKE_ERROR"
        }
    except Exception as e:
        return {
            "has_ssl": False,
            "valid": False,
            "reason": f"Connection failed on port 443: {e}",
            "error_code": "CONNECTION_FAILED"
        }


# Direct WHOIS parser to get Domain Age
def get_whois_raw(domain: str, server: str = "whois.iana.org") -> str:
    """Connects directly to Port 43 WHOIS server to get registration info."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(4.0)
        s.connect((server, 43))
        s.send((domain + "\r\n").encode("utf-8"))
        response = b""
        while True:
            data = s.recv(4096)
            if not data:
                break
            response += data
        s.close()
        return response.decode("latin-1", errors="ignore")
    except Exception:
        return ""

def parse_whois_creation_date(raw_text: str) -> datetime:
    """Parses common creation date formats from WHOIS response text."""
    patterns = [
        (r"(?:Creation Date|created|Created On|Registered on|Date of Creation|Domain Registration Date|Registration Time|created-date|Record created on):\s*([^\r\n]+)", None),
        (r"\[Created Date\]\s*([^\r\n]+)", None),
    ]

    for pat, _ in patterns:
        match = re.search(pat, raw_text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            # Try parsing various formats
            date_formats = [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d-%b-%Y",
                "%Y/%m/%d",
                "%d/%m/%Y",
                "%d.%m.%Y",
                "%Y.%m.%d"
            ]
            for fmt in date_formats:
                try:
                    # Standardize string: extract date part before any timezone offset/filler
                    clean_str = date_str.split(".")[0].split("+")[0].strip()
                    if clean_str.endswith("Z"):
                        clean_str = clean_str[:-1]
                    return datetime.strptime(clean_str, fmt)
                except ValueError:
                    continue
    return None

def check_domain_age_days(domain: str) -> dict:
    """
    Looks up WHOIS details to determine the domain age.
    """
    if not domain or domain.lower() in ["localhost", "127.0.0.1", "example.com"]:
        return {"status": "skip", "age_days": None}

    try:
        # Query IANA first
        raw = get_whois_raw(domain)
        if not raw:
            return {"status": "failed", "age_days": None, "reason": "No response from root WHOIS server"}

        # Check for refer server
        refer_match = re.search(r"refer:\s+(\S+)", raw)
        if refer_match:
            ref_server = refer_match.group(1).strip()
            raw = get_whois_raw(domain, ref_server)

        create_date = parse_whois_creation_date(raw)
        
        if create_date:
            age_days = (datetime.utcnow() - create_date).days
            return {
                "status": "success",
                "creation_date": create_date.strftime("%Y-%m-%d"),
                "age_days": age_days,
                "is_new_domain": age_days < 90  # flag domains under 3 months
            }
        
        return {"status": "failed", "age_days": None, "reason": "Could not parse creation date from WHOIS response"}
    except Exception as e:
        return {"status": "error", "age_days": None, "reason": str(e)}


# Unified Checker Function
def check_url_reputation(url: str) -> dict:
    """
    Performs fully cached reputation verification for a URL (Safe Browsing, SSL, WHOIS age)
    """
    if not url:
        return {}

    parsed = urllib.parse.urlparse(url)
    hostname = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    
    # Try getting from cache
    cache_key = f"reputation:{url}"
    cached_res = cache.get(cache_key)
    if cached_res:
        return cached_res

    # Skip external calls if in offline mode
    if config.OFFLINE_MODE:
        return {
            "mode": "offline",
            "safe_browsing": {"status": "offline", "is_flagged": False},
            "ssl": {"has_ssl": url.lower().startswith("https"), "valid": url.lower().startswith("https")},
            "domain_age": {"status": "offline", "age_days": None}
        }

    # Run check components
    print(f"Checking reputation for {url} (Cache Miss)...")
    
    # Safe Browsing check
    gsb_res = check_google_safe_browsing(url)
    
    # SSL check (only if hostname is available)
    ssl_res = validate_ssl(hostname) if hostname else {"has_ssl": False, "valid": False}
    
    # WHOIS check (get domain base name)
    domain_match = re.search(r"([^.]+\.[^.]+)$", hostname)
    domain = domain_match.group(1) if domain_match else hostname
    whois_res = check_domain_age_days(domain) if domain else {"status": "skip", "age_days": None}
    
    reputation_score = 100
    deductions = []
    
    if gsb_res.get("is_flagged"):
        reputation_score -= 80
        deductions.append(f"Flagged by Google Safe Browsing: {gsb_res.get('threat_type')}")
        
    if not ssl_res.get("has_ssl"):
        # We don't penalize HTTP sites alone if WHOIS shows they are old/reputable (reduce false positives)
        is_new = whois_res.get("is_new_domain", False)
        if is_new:
            reputation_score -= 20
            deductions.append("Missing HTTPS certificate on a new domain (<90 days)")
        else:
            # Moderate penalty if missing HTTPS but domain is old
            reputation_score -= 5
            deductions.append("Missing HTTPS certificate (low penalty since domain is established)")
    else:
        if not ssl_res.get("valid"):
            reputation_score -= 30
            deductions.append(f"Invalid SSL Certificate: {ssl_res.get('reason')}")
            
    if whois_res.get("is_new_domain"):
        reputation_score -= 15
        deductions.append(f"Domain is very new: Registered on {whois_res.get('creation_date')} ({whois_res.get('age_days')} days old)")
        
    reputation_score = max(0, reputation_score)
    
    result = {
        "mode": "online",
        "url": url,
        "hostname": hostname,
        "reputation_score": reputation_score,
        "deductions": deductions,
        "safe_browsing": gsb_res,
        "ssl": ssl_res,
        "domain_age": whois_res,
        "timestamp": time.time()
    }
    
    # Store in cache
    cache.set(cache_key, result)
    return result
