# Quick Start Guide

## 🚀 Setup in 5 Steps

### 1. Install Packages
```bash
pip install -r requirements.txt
```

**Key packages:**
- `fastapi`, `uvicorn` - Web framework
- `transformers`, `torch` - CLIP model for embeddings
- `cloudinary`, `python-dotenv` - Cloud image storage

### 2. Setup Cloudinary (Optional)
Create `.env` file:
```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

Get credentials from: https://cloudinary.com/console

### 3. Add Initial Images
Place images in `photos/` folder:
```
photos/
  ├── logo1.png
  ├── logo2.jpg
  └── ...
```

### 4. Build Initial Database
```bash
python vector.py
```

This will:
- ✅ Generate embeddings for all images
- ✅ Upload to Cloudinary (if configured)
- ✅ Save to `image_embeddings.pkl`

**Output:**
```
Loading CLIP model...
CLIP model loaded successfully
Building vector database...
  ✓ Uploaded to Cloudinary: https://...
✓ Processed: logo1.png
...
✓ Saved 10 embeddings to image_embeddings.pkl
```

### 5. Start Backend
```bash
.\run_https.ps1
```

Or:
```bash
uvicorn main:app --host 127.0.0.1 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

## 📝 How It Works

### Initial Setup (One-time)
1. Run `python vector.py` → Creates `image_embeddings.pkl` with embeddings + Cloudinary URLs

### Adding New Images
- **Via API**: Use `/store` endpoint → Auto uploads to Cloudinary + updates database
- **Bulk**: Add to `photos/` folder → Run `python vector.py` again

### Searching
- Use `/search` endpoint → Returns similar images with Cloudinary URLs

## ✅ Verification

Check if everything worked:
```bash
python -c "import pickle; f=open('image_embeddings.pkl','rb'); emb=pickle.load(f); print(f'✓ {len(emb)} embeddings loaded'); f.close()"
```

## 📚 Full Documentation

See `SETUP_WORKFLOW.md` for detailed instructions.
