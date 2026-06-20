# Phishing Detector - Security Upgrade & API Configuration Guide

This guide describes how to configure, configure, and maintain the upgraded Phishing Detector.

---

## 1. Environment Variables Reference

Configure these environment variables in your local terminal or deployment panel (e.g., Railway, Heroku):

| Variable | Description | Default Value | Example Value |
| :--- | :--- | :--- | :--- |
| `SAFE_BROWSING_API_KEY` | Google Safe Browsing API Key. | None (disabled) | `AIzaSyA1...` |
| `GEMINI_API_KEY` | Gemini API Key for Advanced AI scans. | None (disabled) | `AIzaSyB3...` |
| `PHISH_CACHE_TYPE` | Type of cache to use (`file` or `redis`). | `file` | `redis` |
| `PHISH_CACHE_FILE` | Absolute path to the JSON cache file. | `data/cache.json` | `C:/data/cache.json` |
| `REDIS_URL` | Redis instance URL. | `redis://localhost:6379/0` | `redis://default:pass@redis.railway.internal:6379` |
| `PHISH_CACHE_EXPIRY` | Time-to-live for cache entries in seconds. | `86400` (24h) | `3600` (1h) |
| `PHISH_RATE_LIMIT_MAX` | Max scans allowed per client in rate window. | `100` | `50` |
| `PHISH_OFFLINE_MODE` | Set `true` to force offline pattern matching. | `false` | `true` |

---

## 2. Google Safe Browsing API Configuration

The application uses Google Safe Browsing Lookup API (v4) to test URLs against Google's global malware and social engineering database.

### Setup Instructions
1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Search for the **Safe Browsing API** in the API Library and click **Enable**.
4. Navigate to **APIs & Services** > **Credentials**.
5. Click **Create Credentials** > **API Key**.
6. Set restrictions on your API Key to limit misuse (e.g., restricting it only to the Safe Browsing API).
7. Save the key and export it:
   ```bash
   # On Windows (PowerShell)
   $env:SAFE_BROWSING_API_KEY="your-api-key"
   
   # On Linux/macOS
   export SAFE_BROWSING_API_KEY="your-api-key"
   ```

### Quotas & Costs
- **Cost**: The Safe Browsing Lookup API is **completely free** for non-commercial or low-volume security uses.
- **Default Quota**: 
  - **10,000 requests per day** (can be raised upon request to Google).
  - Rate limits are measured per minute (typically up to ~300 requests per minute).

---

## 3. Caching & Performance Optimization

To prevent API rate limits, domain verification results are cached for **24 hours** by default.

### A. File-Based Caching (Default)
Ideal for local testing and lightweight workloads.
- Cache entries are saved to `data/cache.json`.
- Automatically handles expiry times. No external systems required.

### B. Redis Caching (Production)
Recommended for multi-worker containerized deployments.
- Set `PHISH_CACHE_TYPE=redis`.
- Provide the connection string via `REDIS_URL`.
- Highly fast, memory-mapped storage with automated key eviction (using Redis TTL).

---

## 4. Offline Fallback & Reliability

If external networks fail or keys are unconfigured, the application maintains complete functionality:
1. **Fallback Logic**: If Google Safe Browsing, WHOIS, or SSL validations fail or time out (3.0–5.0 seconds), the app catches exceptions and falls back to standard lexical pattern matching.
2. **Backward Compatibility**: Standard scan results are generated using the local TensorFlow neural network (`phishing_model_combined.h5` or `phishing_model.h5`), scaled via numpy coefficients.
3. **Hybrid Mode**: If APIs are online, their findings adjust the NN score (reducing false positives on verified old domains, increasing on blacklists or newly-registered domains).

---

## 5. Manual False-Positive Reporting

If a user gets an incorrect classification, they can click the **"Report False Positive / Correct Result"** button on the details panel.
- Action: Logs the target URL and email text to a flat dataset file (`data/reported_false_positives.jsonl`).
- Use: Collects targeted data rows for subsequent training iterations to prune overfitting parameters.
