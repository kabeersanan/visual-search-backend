# Setup and Workflow Guide

## 📦 Packages to Install

### Option 1: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Install individually
The key packages you need are:
```bash
# Core dependencies
pip install fastapi uvicorn python-multipart

# ML/AI dependencies
pip install transformers torch torchvision Pillow numpy scipy

# Cloudinary for cloud storage
pip install cloudinary python-dotenv
```

## 🔄 Complete Workflow

### Step 1: Setup Cloudinary (Optional but Recommended)

1. **Create `.env` file** in project root:
```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

2. **Get credentials from** [Cloudinary Dashboard](https://cloudinary.com/console)

> **Note**: If you don't set up Cloudinary, images will still be processed but won't be uploaded to cloud storage.

### Step 2: Prepare Initial Images

Place all your initial brand kit images in the `photos/` folder:
```
photos/
  ├── logo1.png
  ├── logo2.jpg
  ├── icon1.webp
  └── ...
```

### Step 3: Run vector.py to Create Initial Database

This will:
- Generate embeddings for all images in `photos/` folder
- Upload images to Cloudinary (if configured)
- Save embeddings + Cloudinary URLs to `image_embeddings.pkl`

```bash
python vector.py
```

**Output:**
```
Loading CLIP model...
CLIP model loaded successfully
Building vector database...
  ✓ Uploaded to Cloudinary: https://res.cloudinary.com/...
✓ Processed: logo1.png
  ✓ Uploaded to Cloudinary: https://res.cloudinary.com/...
✓ Processed: logo2.jpg
...

✓ Saved 10 embeddings to image_embeddings.pkl
```

### Step 4: Start the Backend Server

```bash
# Using PowerShell (with HTTPS)
.\run_https.ps1

# Or directly with uvicorn
uvicorn main:app --host 127.0.0.1 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

### Step 5: Upload New Images via API

When users upload new images through `/store` endpoint:

1. **Image is uploaded to Cloudinary** automatically
2. **Embedding is generated** and stored
3. **Cloudinary URL is saved** with the embedding
4. **Database is updated** (`image_embeddings.pkl` is regenerated)

**Frontend Example:**
```typescript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('https://localhost:8443/store', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.cloudinary_url); // ✅ Cloudinary URL returned
```

### Step 6: Search for Similar Images

```typescript
const formData = new FormData();
formData.append('file', searchImage);

const response = await fetch('https://localhost:8443/search', {
  method: 'POST',
  body: formData
});

const result = await response.json();
// result.similar_images contains Cloudinary URLs
```

## 📊 Data Structure

### `image_embeddings.pkl` Format

After running `vector.py` or uploading via `/store`, the structure is:

```python
{
  "logo1.png": {
    "embedding": numpy_array([0.1, 0.2, ...]),  # CLIP embedding vector
    "cloudinary_url": "https://res.cloudinary.com/.../brandkit/logo1.png"
  },
  "logo2.jpg": {
    "embedding": numpy_array([0.3, 0.4, ...]),
    "cloudinary_url": "https://res.cloudinary.com/.../brandkit/logo2.jpg"
  }
}
```

## 🔄 When to Re-run vector.py

You only need to run `vector.py` when:
- ✅ **First time setup** - Creating initial database from existing images
- ✅ **Adding many images at once** - Bulk processing existing images
- ✅ **Regenerating database** - After manually editing/adding images to `photos/` folder

You DON'T need to run `vector.py` when:
- ❌ **Users upload via `/store` API** - This handles everything automatically
- ❌ **Single new image** - Use `/store` endpoint instead

## 📝 Complete Example Workflow

### Scenario: Setting up from scratch

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup Cloudinary (create .env file)
# Edit .env with your Cloudinary credentials

# 3. Add initial images to photos/ folder
# Copy logo1.png, logo2.jpg, etc. to photos/

# 4. Build initial database
python vector.py

# 5. Start backend
.\run_https.ps1

# 6. Test search
# Use frontend to search or test via curl/Postman
```

### Scenario: Adding new images

**Option A: Via API (Recommended)**
- Use `/store` endpoint from frontend
- Automatically uploads to Cloudinary and updates database

**Option B: Via vector.py (For bulk)**
- Add images to `photos/` folder
- Run `python vector.py` again
- Will upload new images to Cloudinary and regenerate database

## ⚠️ Important Notes

1. **Cloudinary is optional**: If not configured, embeddings will still work but `cloudinary_url` will be `None`

2. **Database regeneration**: Running `vector.py` will **overwrite** `image_embeddings.pkl`. If you've added images via `/store`, those will be lost unless they're also in the `photos/` folder.

3. **Best practice**: 
   - Use `vector.py` for **initial setup** or **bulk imports**
   - Use `/store` API for **user uploads** and **incremental additions**

4. **Photos folder**: The `photos/` folder is for initial images. After running `vector.py`, images can be deleted from local storage if they're successfully uploaded to Cloudinary.

## 🔍 Verification

After setup, verify everything works:

```bash
# Check embeddings file exists
python -c "import pickle; f=open('image_embeddings.pkl','rb'); emb=pickle.load(f); print(f'Loaded {len(emb)} embeddings'); print(f'First image: {list(emb.keys())[0]}'); print(f'Has Cloudinary URL: {\"cloudinary_url\" in list(emb.values())[0] if isinstance(list(emb.values())[0], dict) else False}'); f.close()"
```

You should see:
- Number of embeddings
- Sample filename
- Whether Cloudinary URLs are stored
