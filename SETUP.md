# Quick Setup Guide

## 1. Install Dependencies

```bash
# Activate your virtual environment first
# Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Note:** This will install:
- FastAPI and Uvicorn (web framework)
- ChromaDB (vector database)
- Sentence Transformers with CLIP (image embeddings)
- PyTorch (ML framework)

## 2. First Run - Model Download

The first time you run the server, it will automatically download the CLIP model (~500MB). This happens automatically but may take a few minutes.

## 3. Start the Server

```bash
# Using the provided script
.\run_https.ps1

# Or manually:
uvicorn main:app --host 127.0.0.1 --port 8443 --ssl-keyfile certs\localhost-key.pem --ssl-certfile certs\localhost.pem --reload
```

## 4. Test the API

### Check health:
```bash
curl https://localhost:8443/health
```

### Add an image:
```bash
curl -X POST "https://localhost:8443/add" \
  -F "file=@your_image.jpg" \
  -k  # Ignore SSL certificate warning in curl
```

### Search for similar images:
```bash
curl -X POST "https://localhost:8443/search" \
  -F "file=@search_image.png" \
  -F "top_k=5" \
  -k
```

## 5. Directory Structure

After setup, you'll have:
```
visual-search-backend/
├── main.py              # FastAPI server
├── vector_db.py         # Vector database service
├── embeddings.py        # Image embedding service
├── requirements.txt     # Dependencies
├── uploads/            # Saved images (auto-created)
├── chroma_db/          # Vector database storage (auto-created)
└── certs/              # SSL certificates
```

## Troubleshooting

### Import Errors
If you get import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model Download Fails
Manually download the model:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32')"
```

### Port Already in Use
Change the port in `run_https.ps1` or use:
```bash
uvicorn main:app --host 127.0.0.1 --port 8444 --ssl-keyfile certs\localhost-key.pem --ssl-certfile certs\localhost.pem
```

### CUDA/GPU Issues
The code will automatically use CPU if GPU is not available. To force CPU, modify `embeddings.py`.
