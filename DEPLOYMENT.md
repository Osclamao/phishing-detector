# Deployment Guide for Phishing Detection Web App

## 🚀 Quick Deploy to Render (Recommended)

### Prerequisites
1. Create a GitHub account (if you don't have one)
2. Create a Render account at https://render.com

### Steps:

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/phishing-detector.git
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to https://render.com/dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name:** phishing-detector
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `cd web_app && gunicorn --bind 0.0.0.0:$PORT wsgi:app`
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment

3. **Access your app:**
   - Your app will be live at: `https://phishing-detector.onrender.com`

---

## 🚂 Alternative: Railway.app

1. Sign up at https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects and deploys (uses Procfile)
5. Get your live URL from the dashboard

---

## 🐍 Alternative: PythonAnywhere (Free Forever)

1. Sign up at https://www.pythonanywhere.com
2. Upload files via "Files" tab
3. Create new web app (Flask)
4. Configure WSGI file to point to your app
5. Install packages: `pip install -r requirements.txt`

**Note:** PythonAnywhere has RAM limits that might affect TensorFlow performance.

---

## 📦 Files Created for Deployment

- `requirements.txt` - Updated with Flask and Gunicorn
- `Procfile` - Instructions for starting your app
- `runtime.txt` - Specifies Python version
- `wsgi.py` - Production entry point

---

## ⚠️ Important Notes

### Model Files
Your model files (`.h5` and `.npy`) are large. Make sure they're pushed to GitHub:
- `phishing_model_combined.h5`
- `scaler_mean_combined.npy`
- `scaler_scale_combined.npy`

### Git Large Files
If your model files are >100MB, use Git LFS:
```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.npy"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

### Environment Variables (if needed)
For sensitive data, use environment variables in Render/Railway dashboard.

---

## 🌐 Custom Domain (Optional)

All platforms support custom domains:
- Render: Settings → Custom Domain
- Railway: Settings → Domains
- PythonAnywhere: Web tab → Add custom domain (paid plan)

---

## 💰 Cost Comparison

| Platform | Free Tier | Limitations |
|----------|-----------|-------------|
| **Render** | Yes | 750 hrs/month, auto-sleep after 15min inactive |
| **Railway** | $5 credit/month | ~500hrs runtime |
| **PythonAnywhere** | Yes forever | 512MB RAM, 1 web app |
| **Vercel** | Yes | Good for lightweight apps only |

---

## 🐛 Troubleshooting

### App crashes on startup:
- Check logs in platform dashboard
- Verify all model files are uploaded
- Check Python version compatibility

### Out of memory:
- TensorFlow models are large (~100MB+)
- Consider using Railway or Render (more RAM)
- Or split model loading on-demand

### Slow cold starts:
- Normal for free tiers (30-60 seconds)
- Render sleeps after inactivity
- Consider paid tier for always-on service
